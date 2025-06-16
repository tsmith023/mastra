import type { D1Database } from '@cloudflare/workers-types';
import { MessageList } from '@mastra/core/agent';
import type { MetricResult, TestInfo } from '@mastra/core/eval';
import type { StorageThreadType, MastraMessageV1, MastraMessageV2 } from '@mastra/core/memory';
import {
  MastraStorage,
  TABLE_MESSAGES,
  TABLE_THREADS,
  TABLE_WORKFLOW_SNAPSHOT,
  TABLE_EVALS,
  TABLE_TRACES,
} from '@mastra/core/storage';
import type {
  TABLE_NAMES,
  StorageColumn,
  StorageGetMessagesArg,
  EvalRow,
  WorkflowRuns,
  WorkflowRun,
  PaginationInfo,
} from '@mastra/core/storage';
import type { Trace } from '@mastra/core/telemetry';
import type { WorkflowRunState } from '@mastra/core/workflows';
import Cloudflare from 'cloudflare';
import { createSqlBuilder } from './sql-builder';
import type { SqlParam, SqlQueryOptions } from './sql-builder';

/**
 * Configuration for D1 using the REST API
 */
export interface D1Config {
  /** Cloudflare account ID */
  accountId: string;
  /** Cloudflare API token with D1 access */
  apiToken: string;
  /** D1 database ID */
  databaseId: string;
  /** Optional prefix for table names */
  tablePrefix?: string;
}

/**
 * Configuration for D1 using the Workers Binding API
 */
export interface D1WorkersConfig {
  /** D1 database binding from Workers environment */
  binding: D1Database; // D1Database binding from Workers
  /** Optional prefix for table names */
  tablePrefix?: string;
}

/**
 * Combined configuration type supporting both REST API and Workers Binding API
 */
export type D1StoreConfig = D1Config | D1WorkersConfig;

function isArrayOfRecords(value: any): value is Record<string, any>[] {
  return value && Array.isArray(value) && value.length > 0;
}

export class D1Store extends MastraStorage {
  private client?: Cloudflare;
  private accountId?: string;
  private databaseId?: string;
  private binding?: D1Database; // D1Database binding
  private tablePrefix: string;

  /**
   * Creates a new D1Store instance
   * @param config Configuration for D1 access (either REST API or Workers Binding API)
   */
  constructor(config: D1StoreConfig) {
    super({ name: 'D1' });

    if (config.tablePrefix && !/^[a-zA-Z0-9_]*$/.test(config.tablePrefix)) {
      throw new Error('Invalid tablePrefix: only letters, numbers, and underscores are allowed.');
    }

    this.tablePrefix = config.tablePrefix || '';

    // Determine which API to use based on provided config
    if ('binding' in config) {
      if (!config.binding) {
        throw new Error('D1 binding is required when using Workers Binding API');
      }
      this.binding = config.binding;
      this.logger.info('Using D1 Workers Binding API');
    } else {
      if (!config.accountId || !config.databaseId || !config.apiToken) {
        throw new Error('accountId, databaseId, and apiToken are required when using REST API');
      }
      this.accountId = config.accountId;
      this.databaseId = config.databaseId;
      this.client = new Cloudflare({
        apiToken: config.apiToken,
      });
      this.logger.info('Using D1 REST API');
    }
  }

  // Helper method to get the full table name with prefix
  private getTableName(tableName: TABLE_NAMES): string {
    return `${this.tablePrefix}${tableName}`;
  }

  private formatSqlParams(params: SqlParam[]): string[] {
    return params.map(p => (p === undefined || p === null ? null : p) as string);
  }

  private async executeWorkersBindingQuery({
    sql,
    params = [],
    first = false,
  }: SqlQueryOptions): Promise<Record<string, any>[] | Record<string, any> | null> {
    // Ensure binding is defined
    if (!this.binding) {
      throw new Error('Workers binding is not configured');
    }

    try {
      const statement = this.binding.prepare(sql);
      const formattedParams = this.formatSqlParams(params);

      // Bind parameters if any
      let result;
      if (formattedParams.length > 0) {
        if (first) {
          result = await statement.bind(...formattedParams).first();
          if (!result) return null;

          return result;
        } else {
          result = await statement.bind(...formattedParams).all();
          const results = result.results || [];

          // Include metadata for debugging if available
          if (result.meta) {
            this.logger.debug('Query metadata', { meta: result.meta });
          }

          return results;
        }
      } else {
        if (first) {
          result = await statement.first();
          if (!result) return null;

          return result;
        } else {
          result = await statement.all();
          const results = result.results || [];

          // Include metadata for debugging if available
          if (result.meta) {
            this.logger.debug('Query metadata', { meta: result.meta });
          }

          return results;
        }
      }
    } catch (workerError: any) {
      this.logger.error('Workers Binding API error', {
        message: workerError instanceof Error ? workerError.message : String(workerError),
        sql,
      });
      throw new Error(`D1 Workers API error: ${workerError.message}`);
    }
  }

  private async executeRestQuery({
    sql,
    params = [],
    first = false,
  }: SqlQueryOptions): Promise<Record<string, any>[] | Record<string, any> | null> {
    // Ensure required properties are defined
    if (!this.client || !this.accountId || !this.databaseId) {
      throw new Error('Missing required REST API configuration');
    }

    try {
      const response = await this.client.d1.database.query(this.databaseId, {
        account_id: this.accountId,
        sql: sql,
        params: this.formatSqlParams(params),
      });

      const result = response.result || [];
      const results = result.flatMap(r => r.results || []);

      // If first=true, return only the first result
      if (first) {
        const firstResult = isArrayOfRecords(results) && results.length > 0 ? results[0] : null;
        if (!firstResult) return null;

        return firstResult;
      }

      return results;
    } catch (restError: any) {
      this.logger.error('REST API error', {
        message: restError instanceof Error ? restError.message : String(restError),
        sql,
      });
      throw new Error(`D1 REST API error: ${restError.message}`);
    }
  }

  /**
   * Execute a SQL query against the D1 database
   * @param options Query options including SQL, parameters, and whether to return only the first result
   * @returns Query results as an array or a single object if first=true
   */
  private async executeQuery(options: SqlQueryOptions): Promise<Record<string, any>[] | Record<string, any> | null> {
    const { sql, params = [], first = false } = options;

    try {
      this.logger.debug('Executing SQL query', { sql, params, first });

      if (this.binding) {
        // Use Workers Binding API
        return this.executeWorkersBindingQuery({ sql, params, first });
      } else if (this.client && this.accountId && this.databaseId) {
        // Use REST API
        return this.executeRestQuery({ sql, params, first });
      } else {
        throw new Error('No valid D1 configuration provided');
      }
    } catch (error: any) {
      this.logger.error('Error executing SQL query', {
        message: error instanceof Error ? error.message : String(error),
        sql,
        params,
        first,
      });
      throw new Error(`D1 query error: ${error.message}`);
    }
  }

  // Helper to get existing table columns
  private async getTableColumns(tableName: string): Promise<{ name: string; type: string }[]> {
    try {
      const sql = `PRAGMA table_info(${tableName})`;
      const result = await this.executeQuery({ sql, params: [] });

      if (!result || !Array.isArray(result)) {
        return [];
      }

      return result.map((row: any) => ({
        name: row.name,
        type: row.type,
      }));
    } catch (error) {
      this.logger.error(`Error getting table columns for ${tableName}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }

  // Helper to serialize objects to JSON strings
  private serializeValue(value: any): any {
    if (value === null || value === undefined) return null;

    if (value instanceof Date) {
      return this.serializeDate(value);
    }

    if (typeof value === 'object') {
      return JSON.stringify(value);
    }

    return value;
  }

  // Helper to deserialize JSON strings to objects
  private deserializeValue(value: any, type?: string): any {
    if (value === null || value === undefined) return null;

    if (type === 'date' && typeof value === 'string') {
      return new Date(value);
    }

    if (type === 'jsonb' && typeof value === 'string') {
      try {
        return JSON.parse(value) as Record<string, any>;
      } catch {
        return value;
      }
    }

    if (typeof value === 'string' && (value.startsWith('{') || value.startsWith('['))) {
      try {
        return JSON.parse(value) as Record<string, any>;
      } catch {
        return value;
      }
    }

    return value;
  }

  protected getSqlType(type: StorageColumn['type']): string {
    switch (type) {
      case 'bigint':
        return 'INTEGER'; // SQLite uses INTEGER for all integer sizes
      case 'jsonb':
        return 'TEXT'; // Store JSON as TEXT in SQLite
      default:
        return super.getSqlType(type);
    }
  }

  async createTable({
    tableName,
    schema,
  }: {
    tableName: TABLE_NAMES;
    schema: Record<string, StorageColumn>;
  }): Promise<void> {
    const fullTableName = this.getTableName(tableName);

    // Build SQL columns from schema
    const columnDefinitions = Object.entries(schema).map(([colName, colDef]) => {
      const type = this.getSqlType(colDef.type);
      const nullable = colDef.nullable === false ? 'NOT NULL' : '';
      const primaryKey = colDef.primaryKey ? 'PRIMARY KEY' : '';
      return `${colName} ${type} ${nullable} ${primaryKey}`.trim();
    });

    // Add table-level constraints if needed
    const tableConstraints: string[] = [];
    if (tableName === TABLE_WORKFLOW_SNAPSHOT) {
      tableConstraints.push('UNIQUE (workflow_name, run_id)');
    }

    const query = createSqlBuilder().createTable(fullTableName, columnDefinitions, tableConstraints);

    const { sql, params } = query.build();

    try {
      await this.executeQuery({ sql, params });
      this.logger.debug(`Created table ${fullTableName}`);
    } catch (error) {
      this.logger.error(`Error creating table ${fullTableName}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      throw new Error(`Failed to create table ${fullTableName}: ${error}`);
    }
  }

  /**
   * Alters table schema to add columns if they don't exist
   * @param tableName Name of the table
   * @param schema Schema of the table
   * @param ifNotExists Array of column names to add if they don't exist
   */
  async alterTable({
    tableName,
    schema,
    ifNotExists,
  }: {
    tableName: TABLE_NAMES;
    schema: Record<string, StorageColumn>;
    ifNotExists: string[];
  }): Promise<void> {
    const fullTableName = this.getTableName(tableName);

    try {
      // Check which columns already exist
      const existingColumns = await this.getTableColumns(fullTableName);
      const existingColumnNames = new Set(existingColumns.map(col => col.name.toLowerCase()));

      // Add missing columns from ifNotExists list
      for (const columnName of ifNotExists) {
        if (!existingColumnNames.has(columnName.toLowerCase()) && schema[columnName]) {
          const columnDef = schema[columnName];
          const sqlType = this.getSqlType(columnDef.type);
          const nullable = columnDef.nullable === false ? 'NOT NULL' : '';
          const defaultValue = columnDef.nullable === false ? this.getDefaultValue(columnDef.type) : '';

          const alterSql =
            `ALTER TABLE ${fullTableName} ADD COLUMN ${columnName} ${sqlType} ${nullable} ${defaultValue}`.trim();

          await this.executeQuery({ sql: alterSql, params: [] });
          this.logger.debug(`Added column ${columnName} to table ${fullTableName}`);
        }
      }
    } catch (error) {
      this.logger.error(`Error altering table ${fullTableName}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      throw new Error(`Failed to alter table ${fullTableName}: ${error}`);
    }
  }

  async clearTable({ tableName }: { tableName: TABLE_NAMES }): Promise<void> {
    const fullTableName = this.getTableName(tableName);

    try {
      const query = createSqlBuilder().delete(fullTableName);

      const { sql, params } = query.build();
      await this.executeQuery({ sql, params });
      this.logger.debug(`Cleared table ${fullTableName}`);
    } catch (error) {
      this.logger.error(`Error clearing table ${fullTableName}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      throw new Error(`Failed to clear table ${fullTableName}: ${error}`);
    }
  }

  private async processRecord(record: Record<string, any>): Promise<Record<string, any>> {
    const processedRecord: Record<string, any> = {};
    for (const [key, value] of Object.entries(record)) {
      processedRecord[key] = this.serializeValue(value);
    }
    return processedRecord;
  }

  async insert({ tableName, record }: { tableName: TABLE_NAMES; record: Record<string, any> }): Promise<void> {
    const fullTableName = this.getTableName(tableName);

    // Process record for SQL insertion
    const processedRecord = await this.processRecord(record);

    // Extract columns and values
    const columns = Object.keys(processedRecord);
    const values = Object.values(processedRecord);

    // Build the INSERT query
    const query = createSqlBuilder().insert(fullTableName, columns, values);

    const { sql, params } = query.build();

    try {
      await this.executeQuery({ sql, params });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error inserting into ${fullTableName}:`, { message });
      throw new Error(`Failed to insert into ${fullTableName}: ${error}`);
    }
  }

  async load<R>({ tableName, keys }: { tableName: TABLE_NAMES; keys: Record<string, string> }): Promise<R | null> {
    const fullTableName = this.getTableName(tableName);

    const query = createSqlBuilder().select('*').from(fullTableName);

    // Add WHERE conditions for each key
    let firstKey = true;
    for (const [key, value] of Object.entries(keys)) {
      if (firstKey) {
        query.where(`${key} = ?`, value);
        firstKey = false;
      } else {
        query.andWhere(`${key} = ?`, value);
      }
    }

    query.limit(1);
    const { sql, params } = query.build();

    try {
      const result = await this.executeQuery({ sql, params, first: true });

      if (!result) return null;

      // Process result to handle JSON fields
      const processedResult: Record<string, any> = {};

      for (const [key, value] of Object.entries(result)) {
        processedResult[key] = this.deserializeValue(value);
      }

      return processedResult as R;
    } catch (error) {
      this.logger.error(`Error loading from ${fullTableName}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  async getThreadById({ threadId }: { threadId: string }): Promise<StorageThreadType | null> {
    const thread = await this.load<StorageThreadType>({
      tableName: TABLE_THREADS,
      keys: { id: threadId },
    });

    if (!thread) return null;

    try {
      return {
        ...thread,
        createdAt: this.ensureDate(thread.createdAt) as Date,
        updatedAt: this.ensureDate(thread.updatedAt) as Date,
        metadata:
          typeof thread.metadata === 'string'
            ? (JSON.parse(thread.metadata || '{}') as Record<string, any>)
            : thread.metadata || {},
      };
    } catch (error) {
      this.logger.error(`Error processing thread ${threadId}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  /**
   * @deprecated use getThreadsByResourceIdPaginated instead
   */
  async getThreadsByResourceId({ resourceId }: { resourceId: string }): Promise<StorageThreadType[]> {
    const fullTableName = this.getTableName(TABLE_THREADS);

    try {
      const query = createSqlBuilder().select('*').from(fullTableName).where('resourceId = ?', resourceId);

      const { sql, params } = query.build();
      const results = await this.executeQuery({ sql, params });

      return (isArrayOfRecords(results) ? results : []).map((thread: any) => ({
        ...thread,
        createdAt: this.ensureDate(thread.createdAt) as Date,
        updatedAt: this.ensureDate(thread.updatedAt) as Date,
        metadata:
          typeof thread.metadata === 'string'
            ? (JSON.parse(thread.metadata || '{}') as Record<string, any>)
            : thread.metadata || {},
      }));
    } catch (error) {
      this.logger.error(`Error getting threads by resourceId ${resourceId}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }

  public async getThreadsByResourceIdPaginated(args: {
    resourceId: string;
    page: number;
    perPage: number;
  }): Promise<PaginationInfo & { threads: StorageThreadType[] }> {
    const { resourceId, page, perPage } = args;
    const fullTableName = this.getTableName(TABLE_THREADS);

    const mapRowToStorageThreadType = (row: Record<string, any>): StorageThreadType => ({
      ...(row as StorageThreadType),
      createdAt: this.ensureDate(row.createdAt) as Date,
      updatedAt: this.ensureDate(row.updatedAt) as Date,
      metadata:
        typeof row.metadata === 'string'
          ? (JSON.parse(row.metadata || '{}') as Record<string, any>)
          : row.metadata || {},
    });

    const countQuery = createSqlBuilder().count().from(fullTableName).where('resourceId = ?', resourceId);
    const countResult = (await this.executeQuery(countQuery.build())) as { count: number }[];
    const total = Number(countResult?.[0]?.count ?? 0);

    const selectQuery = createSqlBuilder()
      .select('*')
      .from(fullTableName)
      .where('resourceId = ?', resourceId)
      .orderBy('createdAt', 'DESC')
      .limit(perPage)
      .offset(page * perPage);

    const results = (await this.executeQuery(selectQuery.build())) as Record<string, any>[];
    const threads = results.map(mapRowToStorageThreadType);

    return {
      threads,
      total,
      page,
      perPage,
      hasMore: page * perPage + threads.length < total,
    };
  }

  async saveThread({ thread }: { thread: StorageThreadType }): Promise<StorageThreadType> {
    const fullTableName = this.getTableName(TABLE_THREADS);

    // Prepare the record for SQL insertion
    const threadToSave = {
      id: thread.id,
      resourceId: thread.resourceId,
      title: thread.title,
      metadata: thread.metadata ? JSON.stringify(thread.metadata) : null,
      createdAt: thread.createdAt,
      updatedAt: thread.updatedAt,
    };

    // Process record for SQL insertion
    const processedRecord = await this.processRecord(threadToSave);

    const columns = Object.keys(processedRecord);
    const values = Object.values(processedRecord);

    // Specify which columns to update on conflict (all except id)
    const updateMap: Record<string, string> = {
      resourceId: 'excluded.resourceId',
      title: 'excluded.title',
      metadata: 'excluded.metadata',
      createdAt: 'excluded.createdAt',
      updatedAt: 'excluded.updatedAt',
    };

    // Use the new insert method with ON CONFLICT
    const query = createSqlBuilder().insert(fullTableName, columns, values, ['id'], updateMap);

    const { sql, params } = query.build();

    try {
      await this.executeQuery({ sql, params });
      return thread;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error saving thread to ${fullTableName}:`, { message });
      throw error;
    }
  }

  async updateThread({
    id,
    title,
    metadata,
  }: {
    id: string;
    title: string;
    metadata: Record<string, unknown>;
  }): Promise<StorageThreadType> {
    const thread = await this.getThreadById({ threadId: id });
    if (!thread) {
      throw new Error(`Thread ${id} not found`);
    }
    const fullTableName = this.getTableName(TABLE_THREADS);

    const mergedMetadata = {
      ...(typeof thread.metadata === 'string' ? JSON.parse(thread.metadata) : thread.metadata),
      ...(metadata as Record<string, any>),
    };

    const columns = ['title', 'metadata', 'updatedAt'];
    const values = [title, JSON.stringify(mergedMetadata), new Date().toISOString()];

    const query = createSqlBuilder().update(fullTableName, columns, values).where('id = ?', id);

    const { sql, params } = query.build();

    try {
      await this.executeQuery({ sql, params });

      return {
        ...thread,
        title,
        metadata: {
          ...(typeof thread.metadata === 'string' ? JSON.parse(thread.metadata) : thread.metadata),
          ...(metadata as Record<string, any>),
        },
        updatedAt: new Date(),
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Error updating thread:', { message });
      throw error;
    }
  }

  async deleteThread({ threadId }: { threadId: string }): Promise<void> {
    const fullTableName = this.getTableName(TABLE_THREADS);

    try {
      // Delete the thread
      const deleteThreadQuery = createSqlBuilder().delete(fullTableName).where('id = ?', threadId);

      const { sql: threadSql, params: threadParams } = deleteThreadQuery.build();
      await this.executeQuery({ sql: threadSql, params: threadParams });

      // Also delete associated messages
      const messagesTableName = this.getTableName(TABLE_MESSAGES);
      const deleteMessagesQuery = createSqlBuilder().delete(messagesTableName).where('thread_id = ?', threadId);

      const { sql: messagesSql, params: messagesParams } = deleteMessagesQuery.build();
      await this.executeQuery({ sql: messagesSql, params: messagesParams });
    } catch (error) {
      this.logger.error(`Error deleting thread ${threadId}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      throw new Error(`Failed to delete thread ${threadId}: ${error}`);
    }
  }

  async saveMessages(args: { messages: MastraMessageV1[]; format?: undefined | 'v1' }): Promise<MastraMessageV1[]>;
  async saveMessages(args: { messages: MastraMessageV2[]; format: 'v2' }): Promise<MastraMessageV2[]>;
  async saveMessages(args: {
    messages: MastraMessageV1[] | MastraMessageV2[];
    format?: undefined | 'v1' | 'v2';
  }): Promise<MastraMessageV2[] | MastraMessageV1[]> {
    const { messages, format = 'v1' } = args;
    if (messages.length === 0) return [];

    try {
      const now = new Date();
      const threadId = messages[0]?.threadId;

      // Validate all messages before insert
      for (const [i, message] of messages.entries()) {
        if (!message.id) throw new Error(`Message at index ${i} missing id`);
        if (!message.threadId) throw new Error(`Message at index ${i} missing threadId`);
        if (!message.content) throw new Error(`Message at index ${i} missing content`);
        if (!message.role) throw new Error(`Message at index ${i} missing role`);
        const thread = await this.getThreadById({ threadId: message.threadId });
        if (!thread) {
          throw new Error(`Thread ${message.threadId} not found`);
        }
      }

      // Prepare all messages for insertion (set timestamps, thread_id, etc.)
      const messagesToInsert = messages.map(message => {
        const createdAt = message.createdAt ? new Date(message.createdAt) : now;
        return {
          id: message.id,
          thread_id: message.threadId,
          content: typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
          createdAt: createdAt.toISOString(),
          role: message.role,
          type: message.type || 'v2',
          resourceId: message.resourceId,
        };
      });

      // Insert messages and update thread's updatedAt in parallel
      await Promise.all([
        this.batchInsert({
          tableName: TABLE_MESSAGES,
          records: messagesToInsert,
        }),
        // Update thread's updatedAt timestamp
        this.executeQuery({
          sql: `UPDATE ${this.getTableName(TABLE_THREADS)} SET updatedAt = ? WHERE id = ?`,
          params: [now.toISOString(), threadId],
        }),
      ]);

      this.logger.debug(`Saved ${messages.length} messages`);
      const list = new MessageList().add(messages, 'memory');
      if (format === `v2`) return list.get.all.v2();
      return list.get.all.v1();
    } catch (error) {
      this.logger.error('Error saving messages:', { message: error instanceof Error ? error.message : String(error) });
      throw error;
    }
  }

  private async _getIncludedMessages(threadId: string, selectBy: StorageGetMessagesArg['selectBy']) {
    const include = selectBy?.include;
    if (!include) return null;
    // Build context parameters
    const prevMax = Math.max(...include.map(i => i.withPreviousMessages || 0));
    const nextMax = Math.max(...include.map(i => i.withNextMessages || 0));
    const includeIds = include.map(i => i.id);

    // CTE with ROW_NUMBER for context fetching
    const sql = `
        WITH ordered_messages AS (
          SELECT
            *,
            ROW_NUMBER() OVER (ORDER BY createdAt DESC) AS row_num
          FROM ${this.getTableName(TABLE_MESSAGES)}
          WHERE thread_id = ?
        )
        SELECT
          m.id,
          m.content,
          m.role,
          m.type,
          m.createdAt,
          m.thread_id AS threadId
        FROM ordered_messages m
        WHERE m.id IN (${includeIds.map(() => '?').join(',')})
        OR EXISTS (
          SELECT 1 FROM ordered_messages target
          WHERE target.id IN (${includeIds.map(() => '?').join(',')})
          AND (
            (m.row_num <= target.row_num + ? AND m.row_num > target.row_num)
            OR
            (m.row_num >= target.row_num - ? AND m.row_num < target.row_num)
          )
        )
        ORDER BY m.createdAt DESC
      `;
    const params = [
      threadId,
      ...includeIds, // for m.id IN (...)
      ...includeIds, // for target.id IN (...)
      prevMax,
      nextMax,
    ];
    const messages = await this.executeQuery({ sql, params });
    return messages;
  }

  /**
   * @deprecated use getMessagesPaginated instead
   */
  public async getMessages(args: StorageGetMessagesArg & { format?: 'v1' }): Promise<MastraMessageV1[]>;
  public async getMessages(args: StorageGetMessagesArg & { format: 'v2' }): Promise<MastraMessageV2[]>;
  public async getMessages({
    threadId,
    selectBy,
    format,
  }: StorageGetMessagesArg & { format?: 'v1' | 'v2' }): Promise<MastraMessageV1[] | MastraMessageV2[]> {
    const fullTableName = this.getTableName(TABLE_MESSAGES);
    const limit = typeof selectBy?.last === 'number' ? selectBy.last : 40;
    const include = selectBy?.include || [];
    const messages: any[] = [];

    try {
      if (include.length) {
        const includeResult = await this._getIncludedMessages(threadId, selectBy);
        if (Array.isArray(includeResult)) messages.push(...includeResult);
      }

      // Exclude already fetched ids
      const excludeIds = messages.map(m => m.id);
      const query = createSqlBuilder()
        .select(['id', 'content', 'role', 'type', 'createdAt', 'thread_id AS threadId'])
        .from(fullTableName)
        .where('thread_id = ?', threadId);

      if (excludeIds.length > 0) {
        query.andWhere(`id NOT IN (${excludeIds.map(() => '?').join(',')})`, ...excludeIds);
      }

      query.orderBy('createdAt', 'DESC').limit(limit);

      const { sql, params } = query.build();

      const result = await this.executeQuery({ sql, params });

      if (Array.isArray(result)) messages.push(...result);

      // Sort by creation time to ensure proper order
      messages.sort((a, b) => {
        const aRecord = a as Record<string, any>;
        const bRecord = b as Record<string, any>;
        const timeA = new Date(aRecord.createdAt as string).getTime();
        const timeB = new Date(bRecord.createdAt as string).getTime();
        return timeA - timeB;
      });

      // Parse message content
      const processedMessages = messages.map(message => {
        const processedMsg: Record<string, any> = {};

        for (const [key, value] of Object.entries(message)) {
          if (key === `type` && value === `v2`) continue;
          processedMsg[key] = this.deserializeValue(value);
        }

        return processedMsg;
      });
      this.logger.debug(`Retrieved ${messages.length} messages for thread ${threadId}`);
      const list = new MessageList().add(processedMessages as MastraMessageV1[] | MastraMessageV2[], 'memory');
      if (format === `v2`) return list.get.all.v2();
      return list.get.all.v1();
    } catch (error) {
      this.logger.error('Error retrieving messages for thread', {
        threadId,
        message: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }

  public async getMessagesPaginated({
    threadId,
    selectBy,
    format,
  }: StorageGetMessagesArg & { format?: 'v1' | 'v2' }): Promise<
    PaginationInfo & { messages: MastraMessageV1[] | MastraMessageV2[] }
  > {
    const { dateRange, page = 0, perPage = 40 } = selectBy?.pagination || {};
    const { start: fromDate, end: toDate } = dateRange || {};

    const fullTableName = this.getTableName(TABLE_MESSAGES);
    const messages: any[] = [];

    if (selectBy?.include?.length) {
      const includeResult = await this._getIncludedMessages(threadId, selectBy);
      if (Array.isArray(includeResult)) messages.push(...includeResult);
    }

    const countQuery = createSqlBuilder().count().from(fullTableName).where('thread_id = ?', threadId);

    if (fromDate) {
      countQuery.andWhere('createdAt >= ?', this.serializeDate(fromDate));
    }
    if (toDate) {
      countQuery.andWhere('createdAt <= ?', this.serializeDate(toDate));
    }

    const countResult = (await this.executeQuery(countQuery.build())) as { count: number }[];
    const total = Number(countResult[0]?.count ?? 0);

    const query = createSqlBuilder()
      .select(['id', 'content', 'role', 'type', 'createdAt', 'thread_id AS threadId'])
      .from(fullTableName)
      .where('thread_id = ?', threadId);

    if (fromDate) {
      query.andWhere('createdAt >= ?', this.serializeDate(fromDate));
    }
    if (toDate) {
      query.andWhere('createdAt <= ?', this.serializeDate(toDate));
    }

    query
      .orderBy('createdAt', 'DESC')
      .limit(perPage)
      .offset(page * perPage);

    const results = (await this.executeQuery(query.build())) as any[];
    const list = new MessageList().add(results as MastraMessageV1[] | MastraMessageV2[], 'memory');
    messages.push(...(format === `v2` ? list.get.all.v2() : list.get.all.v1()));

    return {
      messages,
      total,
      page,
      perPage,
      hasMore: page * perPage + messages.length < total,
    };
  }

  async persistWorkflowSnapshot({
    workflowName,
    runId,
    snapshot,
  }: {
    workflowName: string;
    runId: string;
    snapshot: WorkflowRunState;
  }): Promise<void> {
    const fullTableName = this.getTableName(TABLE_WORKFLOW_SNAPSHOT);
    const now = new Date().toISOString();

    const currentSnapshot = await this.load({
      tableName: TABLE_WORKFLOW_SNAPSHOT,
      keys: { workflow_name: workflowName, run_id: runId },
    });

    const persisting = currentSnapshot
      ? {
          ...currentSnapshot,
          snapshot: JSON.stringify(snapshot),
          updatedAt: now,
        }
      : {
          workflow_name: workflowName,
          run_id: runId,
          snapshot: snapshot as Record<string, any>,
          createdAt: now,
          updatedAt: now,
        };

    // Process record for SQL insertion
    const processedRecord = await this.processRecord(persisting);

    const columns = Object.keys(processedRecord);
    const values = Object.values(processedRecord);

    // Specify which columns to update on conflict (all except PKs)
    const updateMap: Record<string, string> = {
      snapshot: 'excluded.snapshot',
      updatedAt: 'excluded.updatedAt',
    };

    this.logger.debug('Persisting workflow snapshot', { workflowName, runId });

    // Use the new insert method with ON CONFLICT
    const query = createSqlBuilder().insert(fullTableName, columns, values, ['workflow_name', 'run_id'], updateMap);

    const { sql, params } = query.build();

    try {
      await this.executeQuery({ sql, params });
    } catch (error) {
      this.logger.error('Error persisting workflow snapshot:', {
        message: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  async loadWorkflowSnapshot(params: { workflowName: string; runId: string }): Promise<WorkflowRunState | null> {
    const { workflowName, runId } = params;

    this.logger.debug('Loading workflow snapshot', { workflowName, runId });

    const d = await this.load<{ snapshot: unknown }>({
      tableName: TABLE_WORKFLOW_SNAPSHOT,
      keys: {
        workflow_name: workflowName,
        run_id: runId,
      },
    });

    return d ? (d.snapshot as WorkflowRunState) : null;
  }

  /**
   * Insert multiple records in a batch operation
   * @param tableName The table to insert into
   * @param records The records to insert
   */
  async batchInsert({ tableName, records }: { tableName: TABLE_NAMES; records: Record<string, any>[] }): Promise<void> {
    if (records.length === 0) return;

    const fullTableName = this.getTableName(tableName);

    try {
      // Process records in batches for better performance
      const batchSize = 50; // Adjust based on performance testing

      for (let i = 0; i < records.length; i += batchSize) {
        const batch = records.slice(i, i + batchSize);

        const recordsToInsert = batch;

        // For bulk insert, we need to determine the columns from the first record
        if (recordsToInsert.length > 0) {
          const firstRecord = recordsToInsert[0];
          // Ensure firstRecord is not undefined before calling Object.keys
          const columns = Object.keys(firstRecord || {});

          // Create a bulk insert statement
          for (const record of recordsToInsert) {
            // Use type-safe approach to extract values
            const values = columns.map(col => {
              if (!record) return null;
              // Safely access the record properties
              const value = typeof col === 'string' ? record[col as keyof typeof record] : null;
              return this.serializeValue(value);
            });

            const query = createSqlBuilder().insert(fullTableName, columns, values);

            const { sql, params } = query.build();
            await this.executeQuery({ sql, params });
          }
        }

        this.logger.debug(
          `Processed batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(records.length / batchSize)}`,
        );
      }

      this.logger.debug(`Successfully batch inserted ${records.length} records into ${tableName}`);
    } catch (error) {
      this.logger.error(`Error batch inserting into ${tableName}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      throw new Error(`Failed to batch insert into ${tableName}: ${error}`);
    }
  }

  /**
   * @deprecated use getTracesPaginated instead
   */
  async getTraces({
    name,
    scope,
    page,
    perPage,
    attributes,
    fromDate,
    toDate,
  }: {
    name?: string;
    scope?: string;
    page: number;
    perPage: number;
    attributes?: Record<string, string>;
    fromDate?: Date;
    toDate?: Date;
  }): Promise<Trace[]> {
    const fullTableName = this.getTableName(TABLE_TRACES);

    try {
      const query = createSqlBuilder().select('*').from(fullTableName).where('1=1');

      if (name) {
        query.andWhere('name LIKE ?', `%${name}%`);
      }

      if (scope) {
        query.andWhere('scope = ?', scope);
      }

      if (attributes && Object.keys(attributes).length > 0) {
        // Handle JSON attribute filtering
        for (const [key, value] of Object.entries(attributes)) {
          query.jsonLike('attributes', key, value);
        }
      }

      if (fromDate) {
        query.andWhere('createdAt >= ?', fromDate instanceof Date ? fromDate.toISOString() : fromDate);
      }

      if (toDate) {
        query.andWhere('createdAt <= ?', toDate instanceof Date ? toDate.toISOString() : toDate);
      }

      query
        .orderBy('startTime', 'DESC')
        .limit(perPage)
        .offset(page * perPage);

      const { sql, params } = query.build();
      const results = await this.executeQuery({ sql, params });

      return isArrayOfRecords(results)
        ? results.map(
            (trace: Record<string, any>) =>
              ({
                ...trace,
                attributes: this.deserializeValue(trace.attributes, 'jsonb'),
                status: this.deserializeValue(trace.status, 'jsonb'),
                events: this.deserializeValue(trace.events, 'jsonb'),
                links: this.deserializeValue(trace.links, 'jsonb'),
                other: this.deserializeValue(trace.other, 'jsonb'),
              }) as Trace,
          )
        : [];
    } catch (error) {
      this.logger.error('Error getting traces:', { message: error instanceof Error ? error.message : String(error) });
      return [];
    }
  }

  public async getTracesPaginated(args: {
    name?: string;
    scope?: string;
    attributes?: Record<string, string>;
    page: number;
    perPage: number;
    fromDate?: Date;
    toDate?: Date;
  }): Promise<PaginationInfo & { traces: Trace[] }> {
    const { name, scope, page, perPage, attributes, fromDate, toDate } = args;
    const fullTableName = this.getTableName(TABLE_TRACES);

    try {
      const dataQuery = createSqlBuilder().select('*').from(fullTableName).where('1=1');
      const countQuery = createSqlBuilder().count().from(fullTableName).where('1=1');

      if (name) {
        dataQuery.andWhere('name LIKE ?', `%${name}%`);
        countQuery.andWhere('name LIKE ?', `%${name}%`);
      }

      if (scope) {
        dataQuery.andWhere('scope = ?', scope);
        countQuery.andWhere('scope = ?', scope);
      }

      if (attributes && Object.keys(attributes).length > 0) {
        for (const [key, value] of Object.entries(attributes)) {
          dataQuery.jsonLike('attributes', key, value);
          countQuery.jsonLike('attributes', key, value);
        }
      }

      if (fromDate) {
        const fromDateStr = fromDate instanceof Date ? fromDate.toISOString() : fromDate;
        dataQuery.andWhere('createdAt >= ?', fromDateStr);
        countQuery.andWhere('createdAt >= ?', fromDateStr);
      }

      if (toDate) {
        const toDateStr = toDate instanceof Date ? toDate.toISOString() : toDate;
        dataQuery.andWhere('createdAt <= ?', toDateStr);
        countQuery.andWhere('createdAt <= ?', toDateStr);
      }

      const countResult = (await this.executeQuery(countQuery.build())) as { count: number }[];
      const total = Number(countResult?.[0]?.count ?? 0);

      dataQuery
        .orderBy('startTime', 'DESC')
        .limit(perPage)
        .offset(page * perPage);

      const results = await this.executeQuery(dataQuery.build());

      const traces = isArrayOfRecords(results)
        ? results.map(
            (trace: Record<string, any>) =>
              ({
                ...trace,
                attributes: this.deserializeValue(trace.attributes, 'jsonb'),
                status: this.deserializeValue(trace.status, 'jsonb'),
                events: this.deserializeValue(trace.events, 'jsonb'),
                links: this.deserializeValue(trace.links, 'jsonb'),
                other: this.deserializeValue(trace.other, 'jsonb'),
              }) as Trace,
          )
        : [];

      return {
        traces,
        total,
        page,
        perPage,
        hasMore: page * perPage + traces.length < total,
      };
    } catch (error) {
      this.logger.error('Error getting traces:', { message: error instanceof Error ? error.message : String(error) });
      return { traces: [], total: 0, page, perPage, hasMore: false };
    }
  }

  /**
   * @deprecated use getEvals instead
   */
  async getEvalsByAgentName(agentName: string, type?: 'test' | 'live'): Promise<EvalRow[]> {
    const fullTableName = this.getTableName(TABLE_EVALS);

    try {
      let query = createSqlBuilder().select('*').from(fullTableName).where('agent_name = ?', agentName);

      // Match LibSQL implementation for type filtering
      if (type === 'test') {
        // For 'test' type: test_info must exist and have a testPath property
        query = query.andWhere("test_info IS NOT NULL AND json_extract(test_info, '$.testPath') IS NOT NULL");
      } else if (type === 'live') {
        // For 'live' type: test_info is NULL or doesn't have a testPath property
        query = query.andWhere("(test_info IS NULL OR json_extract(test_info, '$.testPath') IS NULL)");
      }

      query.orderBy('created_at', 'DESC');

      const { sql, params } = query.build();
      const results = await this.executeQuery({ sql, params });

      return isArrayOfRecords(results)
        ? results.map((row: Record<string, any>) => {
            // Convert snake_case to camelCase for the response
            const result = this.deserializeValue(row.result);
            const testInfo = row.test_info ? this.deserializeValue(row.test_info) : undefined;

            return {
              input: row.input || '',
              output: row.output || '',
              result,
              agentName: row.agent_name || '',
              metricName: row.metric_name || '',
              instructions: row.instructions || '',
              runId: row.run_id || '',
              globalRunId: row.global_run_id || '',
              createdAt: row.created_at || '',
              testInfo,
            };
          })
        : [];
    } catch (error) {
      this.logger.error(`Error getting evals for agent ${agentName}:`, {
        message: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }

  async getEvals(options?: {
    agentName?: string;
    type?: 'test' | 'live';
    page?: number;
    perPage?: number;
    fromDate?: Date;
    toDate?: Date;
  }): Promise<PaginationInfo & { evals: EvalRow[] }> {
    const { agentName, type, page = 0, perPage = 40, fromDate, toDate } = options || {};
    const fullTableName = this.getTableName(TABLE_EVALS);

    const conditions: string[] = [];
    const queryParams: SqlParam[] = [];

    if (agentName) {
      conditions.push(`agent_name = ?`);
      queryParams.push(agentName);
    }

    if (type === 'test') {
      // For SQLite/D1, json_extract is used to query JSON fields
      conditions.push(`(test_info IS NOT NULL AND json_extract(test_info, '$.testPath') IS NOT NULL)`);
    } else if (type === 'live') {
      conditions.push(`(test_info IS NULL OR json_extract(test_info, '$.testPath') IS NULL)`);
    }

    if (fromDate) {
      conditions.push(`createdAt >= ?`);
      queryParams.push(this.serializeDate(fromDate));
    }

    if (toDate) {
      conditions.push(`createdAt <= ?`);
      queryParams.push(this.serializeDate(toDate));
    }

    const countQueryBuilder = createSqlBuilder().count().from(fullTableName);
    if (conditions.length > 0) {
      countQueryBuilder.where(conditions.join(' AND '), ...queryParams);
    }
    const { sql: countSql, params: countParams } = countQueryBuilder.build();

    const countResult = (await this.executeQuery({ sql: countSql, params: countParams, first: true })) as {
      count: number;
    } | null;
    const total = Number(countResult?.count || 0);

    const currentOffset = page * perPage;

    if (total === 0) {
      return {
        evals: [],
        total: 0,
        page,
        perPage,
        hasMore: false,
      };
    }

    const dataQueryBuilder = createSqlBuilder().select('*').from(fullTableName);
    if (conditions.length > 0) {
      dataQueryBuilder.where(conditions.join(' AND '), ...queryParams);
    }
    dataQueryBuilder.orderBy('createdAt', 'DESC').limit(perPage).offset(currentOffset);

    const { sql: dataSql, params: dataParams } = dataQueryBuilder.build();
    const rows = await this.executeQuery({ sql: dataSql, params: dataParams });

    const evals = (isArrayOfRecords(rows) ? rows : []).map((row: Record<string, any>) => {
      const result = this.deserializeValue(row.result);
      const testInfo = row.test_info ? this.deserializeValue(row.test_info) : undefined;

      if (!result || typeof result !== 'object' || !('score' in result)) {
        throw new Error(`Invalid MetricResult format: ${JSON.stringify(result)}`);
      }

      return {
        input: row.input as string,
        output: row.output as string,
        result: result as MetricResult,
        agentName: row.agent_name as string,
        metricName: row.metric_name as string,
        instructions: row.instructions as string,
        testInfo: testInfo as TestInfo,
        globalRunId: row.global_run_id as string,
        runId: row.run_id as string,
        createdAt: row.createdAt as string,
      } as EvalRow;
    });

    const hasMore = currentOffset + evals.length < total;

    return {
      evals,
      total,
      page,
      perPage,
      hasMore,
    };
  }

  private parseWorkflowRun(row: any): WorkflowRun {
    let parsedSnapshot: WorkflowRunState | string = row.snapshot as string;
    if (typeof parsedSnapshot === 'string') {
      try {
        parsedSnapshot = JSON.parse(row.snapshot as string) as WorkflowRunState;
      } catch (e) {
        // If parsing fails, return the raw snapshot string
        console.warn(`Failed to parse snapshot for workflow ${row.workflow_name}: ${e}`);
      }
    }

    return {
      workflowName: row.workflow_name,
      runId: row.run_id,
      snapshot: parsedSnapshot,
      createdAt: this.ensureDate(row.createdAt)!,
      updatedAt: this.ensureDate(row.updatedAt)!,
      resourceId: row.resourceId,
    };
  }

  private async hasColumn(table: string, column: string): Promise<boolean> {
    // For D1/SQLite, use PRAGMA table_info to get column info
    const sql = `PRAGMA table_info(${table});`;
    const result = await this.executeQuery({ sql, params: [] });
    if (!result || !Array.isArray(result)) return false;
    return result.some((col: any) => col.name === column || col.name === column.toLowerCase());
  }

  async getWorkflowRuns({
    workflowName,
    fromDate,
    toDate,
    limit,
    offset,
    resourceId,
  }: {
    workflowName?: string;
    fromDate?: Date;
    toDate?: Date;
    limit?: number;
    offset?: number;
    resourceId?: string;
  } = {}): Promise<WorkflowRuns> {
    const fullTableName = this.getTableName(TABLE_WORKFLOW_SNAPSHOT);
    try {
      const builder = createSqlBuilder().select().from(fullTableName);
      const countBuilder = createSqlBuilder().count().from(fullTableName);

      if (workflowName) builder.whereAnd('workflow_name = ?', workflowName);
      if (resourceId) {
        const hasResourceId = await this.hasColumn(fullTableName, 'resourceId');
        if (hasResourceId) {
          builder.whereAnd('resourceId = ?', resourceId);
          countBuilder.whereAnd('resourceId = ?', resourceId);
        } else {
          console.warn(`[${fullTableName}] resourceId column not found. Skipping resourceId filter.`);
        }
      }
      if (fromDate) {
        builder.whereAnd('createdAt >= ?', fromDate instanceof Date ? fromDate.toISOString() : fromDate);
        countBuilder.whereAnd('createdAt >= ?', fromDate instanceof Date ? fromDate.toISOString() : fromDate);
      }
      if (toDate) {
        builder.whereAnd('createdAt <= ?', toDate instanceof Date ? toDate.toISOString() : toDate);
        countBuilder.whereAnd('createdAt <= ?', toDate instanceof Date ? toDate.toISOString() : toDate);
      }

      builder.orderBy('createdAt', 'DESC');
      if (typeof limit === 'number') builder.limit(limit);
      if (typeof offset === 'number') builder.offset(offset);

      const { sql, params } = builder.build();

      let total = 0;

      if (limit !== undefined && offset !== undefined) {
        const { sql: countSql, params: countParams } = countBuilder.build();
        const countResult = await this.executeQuery({ sql: countSql, params: countParams, first: true });
        total = Number((countResult as Record<string, any>)?.count ?? 0);
      }

      const results = await this.executeQuery({ sql, params });
      const runs = (isArrayOfRecords(results) ? results : []).map((row: any) => this.parseWorkflowRun(row));
      return { runs, total: total || runs.length };
    } catch (error) {
      this.logger.error('Error getting workflow runs:', {
        message: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  async getWorkflowRunById({
    runId,
    workflowName,
  }: {
    runId: string;
    workflowName?: string;
  }): Promise<WorkflowRun | null> {
    const fullTableName = this.getTableName(TABLE_WORKFLOW_SNAPSHOT);
    try {
      const conditions: string[] = [];
      const params: SqlParam[] = [];
      if (runId) {
        conditions.push('run_id = ?');
        params.push(runId);
      }
      if (workflowName) {
        conditions.push('workflow_name = ?');
        params.push(workflowName);
      }
      const whereClause = conditions.length > 0 ? 'WHERE ' + conditions.join(' AND ') : '';
      const sql = `SELECT * FROM ${fullTableName} ${whereClause} ORDER BY createdAt DESC LIMIT 1`;
      const result = await this.executeQuery({ sql, params, first: true });
      if (!result) return null;
      return this.parseWorkflowRun(result);
    } catch (error) {
      this.logger.error('Error getting workflow run by ID:', {
        message: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Close the database connection
   * No explicit cleanup needed for D1 in either REST or Workers Binding mode
   */
  async close(): Promise<void> {
    this.logger.debug('Closing D1 connection');
    // No explicit cleanup needed for D1
  }
}
