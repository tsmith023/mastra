import type { KVNamespace } from '@cloudflare/workers-types';
import { MessageList } from '@mastra/core/agent';
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
  StorageGetTracesArg,
  PaginationInfo,
} from '@mastra/core/storage';
import type { Trace } from '@mastra/core/telemetry';
import type { WorkflowRunState } from '@mastra/core/workflows';
import Cloudflare from 'cloudflare';
import { isWorkersConfig } from './types';
import type { CloudflareStoreConfig, ListOptions, RecordTypes } from './types';

export class CloudflareStore extends MastraStorage {
  private client?: Cloudflare;
  private accountId?: string;
  private namespacePrefix: string;
  private bindings?: Record<TABLE_NAMES, KVNamespace>;

  private validateWorkersConfig(
    config: CloudflareStoreConfig,
  ): asserts config is { bindings: Record<TABLE_NAMES, KVNamespace>; keyPrefix?: string } {
    if (!isWorkersConfig(config)) {
      throw new Error('Invalid Workers API configuration');
    }
    if (!config.bindings) {
      throw new Error('KV bindings are required when using Workers Binding API');
    }

    // Validate all required table bindings exist
    const requiredTables = [TABLE_THREADS, TABLE_MESSAGES, TABLE_WORKFLOW_SNAPSHOT, TABLE_EVALS, TABLE_TRACES] as const;

    for (const table of requiredTables) {
      if (!(table in config.bindings)) {
        throw new Error(`Missing KV binding for table: ${table}`);
      }
    }
  }

  private validateRestConfig(
    config: CloudflareStoreConfig,
  ): asserts config is { accountId: string; apiToken: string; namespacePrefix?: string } {
    if (isWorkersConfig(config)) {
      throw new Error('Invalid REST API configuration');
    }
    if (!config.accountId?.trim()) {
      throw new Error('accountId is required for REST API');
    }
    if (!config.apiToken?.trim()) {
      throw new Error('apiToken is required for REST API');
    }
  }

  constructor(config: CloudflareStoreConfig) {
    super({ name: 'Cloudflare' });

    try {
      if (isWorkersConfig(config)) {
        this.validateWorkersConfig(config);
        this.bindings = config.bindings;
        this.namespacePrefix = config.keyPrefix?.trim() || '';
        this.logger.info('Using Cloudflare KV Workers Binding API');
      } else {
        this.validateRestConfig(config);
        this.accountId = config.accountId.trim();
        this.namespacePrefix = config.namespacePrefix?.trim() || '';
        this.client = new Cloudflare({
          apiToken: config.apiToken.trim(),
        });
        this.logger.info('Using Cloudflare KV REST API');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Failed to initialize CloudflareStore:', { message });
      throw error;
    }
  }

  private getBinding(tableName: TABLE_NAMES) {
    if (!this.bindings) {
      throw new Error(`Cannot use Workers API binding for ${tableName}: Store initialized with REST API configuration`);
    }
    const binding = this.bindings[tableName];
    if (!binding) throw new Error(`No binding found for namespace ${tableName}`);
    return binding;
  }

  private async listNamespaces(): Promise<{
    result: Array<{ id: string; title: string; supports_url_encoding?: boolean }>;
  }> {
    if (this.bindings) {
      return {
        result: Object.keys(this.bindings).map(name => ({
          id: name,
          title: name,
          supports_url_encoding: true,
        })),
      };
    }

    let allNamespaces: Array<Cloudflare.KV.Namespace> = [];
    let currentPage = 1;
    const perPage = 50; // Using 50, max is 100 for namespaces.list
    let morePagesExist = true;

    while (morePagesExist) {
      const response = await this.client!.kv.namespaces.list({
        account_id: this.accountId!,
        page: currentPage,
        per_page: perPage,
      });

      if (response.result) {
        allNamespaces = allNamespaces.concat(response.result);
      }

      morePagesExist = response.result ? response.result.length === perPage : false;

      if (morePagesExist) {
        currentPage++;
      }
    }
    return { result: allNamespaces };
  }

  private async getNamespaceValue(tableName: TABLE_NAMES, key: string) {
    try {
      if (this.bindings) {
        const binding = this.getBinding(tableName);
        const result = await binding.getWithMetadata(key, 'text');
        if (!result) return null;
        return JSON.stringify(result);
      } else {
        const namespaceId = await this.getNamespaceId(tableName);
        const response = await this.client!.kv.namespaces.values.get(namespaceId, key, {
          account_id: this.accountId!,
        });
        return await response.text();
      }
    } catch (error: any) {
      if (error.message && error.message.includes('key not found')) {
        return null;
      }
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Failed to get value for ${tableName} ${key}:`, { message });
      throw error;
    }
  }

  private async putNamespaceValue({
    tableName,
    key,
    value,
    metadata,
  }: {
    tableName: TABLE_NAMES;
    key: string;
    value: string;
    metadata?: any;
  }) {
    try {
      // Ensure consistent serialization
      const serializedValue = this.safeSerialize(value);
      const serializedMetadata = metadata ? this.safeSerialize(metadata) : '';

      if (this.bindings) {
        const binding = this.getBinding(tableName);
        await binding.put(key, serializedValue, { metadata: serializedMetadata });
      } else {
        const namespaceId = await this.getNamespaceId(tableName);
        await this.client!.kv.namespaces.values.update(namespaceId, key, {
          account_id: this.accountId!,
          value: serializedValue,
          metadata: serializedMetadata,
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Failed to put value for ${tableName} ${key}:`, { message });
      throw error;
    }
  }

  private async deleteNamespaceValue(tableName: TABLE_NAMES, key: string) {
    if (this.bindings) {
      const binding = this.getBinding(tableName);
      await binding.delete(key);
    } else {
      const namespaceId = await this.getNamespaceId(tableName);
      await this.client!.kv.namespaces.values.delete(namespaceId, key, {
        account_id: this.accountId!,
      });
    }
  }

  async listNamespaceKeys(tableName: TABLE_NAMES, options?: ListOptions) {
    try {
      if (this.bindings) {
        const binding = this.getBinding(tableName);
        const response = await binding.list({
          limit: options?.limit || 1000,
          prefix: options?.prefix,
        });

        // Convert Workers API response to match REST API format
        return response.keys;
      } else {
        const namespaceId = await this.getNamespaceId(tableName);
        // Use REST API
        const response = await this.client!.kv.namespaces.keys.list(namespaceId, {
          account_id: this.accountId!,
          limit: options?.limit || 1000,
          prefix: options?.prefix,
        });
        return response.result;
      }
    } catch (error: any) {
      this.logger.error(`Failed to list keys for ${tableName}:`, error);
      throw new Error(`Failed to list keys: ${error.message}`);
    }
  }

  private async createNamespaceById(title: string) {
    if (this.bindings) {
      // For Workers API, namespaces are created at deploy time
      // Return a mock response matching REST API shape
      return {
        id: title, // Use title as ID since that's what we need
        title: title,
        supports_url_encoding: true,
      };
    }
    return await this.client!.kv.namespaces.create({
      account_id: this.accountId!,
      title,
    });
  }

  private async getNamespaceIdByName(namespaceName: string): Promise<string | null> {
    try {
      const response = await this.listNamespaces();
      const namespace = response.result.find(ns => ns.title === namespaceName);
      return namespace ? namespace.id : null;
    } catch (error: any) {
      this.logger.error(`Failed to get namespace ID for ${namespaceName}:`, error);
      return null;
    }
  }

  private async createNamespace(namespaceName: string): Promise<string> {
    try {
      const response = await this.createNamespaceById(namespaceName);
      return response.id;
    } catch (error: any) {
      // Check if the error is because it already exists
      if (error.message && error.message.includes('already exists')) {
        // Try to get it again since we know it exists
        const namespaces = await this.listNamespaces();
        const namespace = namespaces.result.find(ns => ns.title === namespaceName);
        if (namespace) return namespace.id;
      }
      this.logger.error('Error creating namespace:', error);
      throw new Error(`Failed to create namespace ${namespaceName}: ${error.message}`);
    }
  }

  private async getOrCreateNamespaceId(namespaceName: string): Promise<string> {
    let namespaceId = await this.getNamespaceIdByName(namespaceName);
    if (!namespaceId) {
      namespaceId = await this.createNamespace(namespaceName);
    }
    return namespaceId;
  }

  private async getNamespaceId(tableName: TABLE_NAMES): Promise<string> {
    const prefix = this.namespacePrefix ? `${this.namespacePrefix}_` : '';

    try {
      const legacyNamespaceId = await this.checkLegacyNamespace(tableName, prefix);
      if (legacyNamespaceId) {
        return legacyNamespaceId;
      }
      return await this.getOrCreateNamespaceId(`${prefix}${tableName}`);
    } catch (error: any) {
      this.logger.error('Error fetching namespace ID:', error);
      throw new Error(`Failed to fetch namespace ID for table ${tableName}: ${error.message}`);
    }
  }

  private LEGACY_NAMESPACE_MAP: Record<string, string> = {
    [TABLE_MESSAGES]: TABLE_THREADS,
    [TABLE_WORKFLOW_SNAPSHOT]: 'mastra_workflows',
    [TABLE_TRACES]: TABLE_EVALS,
  };

  /**
   * There were a few legacy mappings for tables such as
   * - messages -> threads
   * - workflow_snapshot -> mastra_workflows
   * - traces -> evals
   * This has been updated to use dedicated namespaces for each table.
   * In the case of data for a table existing in the legacy namespace, warn the user to migrate to the new namespace.
   *
   * @param tableName The table name to check for legacy data
   * @param prefix The namespace prefix
   * @returns The legacy namespace ID if data exists; otherwise, null
   */
  private async checkLegacyNamespace(tableName: TABLE_NAMES, prefix: string): Promise<string | null> {
    const legacyNamespaceBase = this.LEGACY_NAMESPACE_MAP[tableName];

    // 1. If legacy mapping exists, check for legacy data
    if (legacyNamespaceBase) {
      const legacyNamespace = `${prefix}${legacyNamespaceBase}`;
      const keyPrefix = this.namespacePrefix ? `${this.namespacePrefix}:` : '';
      const prefixKey = `${keyPrefix}${tableName}:`;
      const legacyId = await this.getNamespaceIdByName(legacyNamespace);
      if (legacyId) {
        // Check for any keys for this table in the legacy namespace
        const response = await this.client!.kv.namespaces.keys.list(legacyId, {
          account_id: this.accountId!,
          prefix: prefixKey,
        });
        const keys = response.result;
        const hasTableData = keys.length > 0;
        if (hasTableData) {
          this.logger.warn(
            `Using legacy namespace "${legacyNamespace}" for ${tableName}. Consider migrating to a dedicated namespace "${prefix}${tableName}".`,
          );
          return legacyId;
        }
      }
    }
    return null;
  }

  /**
   * Helper to safely serialize data for KV storage
   */
  private safeSerialize(data: any): string {
    return typeof data === 'string' ? data : JSON.stringify(data);
  }

  /**
   * Helper to safely parse data from KV storage
   */
  private safeParse(text: string | null): any {
    if (!text) return null;
    try {
      const data = JSON.parse(text);
      // If we got an object with a value property that's a string, try to parse that too
      if (data && typeof data === 'object' && 'value' in data) {
        if (typeof data.value === 'string') {
          try {
            return JSON.parse(data.value);
          } catch {
            // If value is a string but not JSON, return as is
            return data.value;
          }
        }
        return null;
      }
      return data;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Failed to parse text:', { message, text });
      return null;
    }
  }

  private async putKV({
    tableName,
    key,
    value,
    metadata,
  }: {
    tableName: TABLE_NAMES;
    key: string;
    value: any;
    metadata?: any;
  }): Promise<void> {
    try {
      await this.putNamespaceValue({ tableName, key, value, metadata });
    } catch (error: any) {
      this.logger.error(`Failed to put KV value for ${tableName}:${key}:`, error);
      throw new Error(`Failed to put KV value: ${error.message}`);
    }
  }

  private async getKV(tableName: TABLE_NAMES, key: string): Promise<any> {
    try {
      const text = await this.getNamespaceValue(tableName, key);
      return this.safeParse(text);
    } catch (error: any) {
      this.logger.error(`Failed to get KV value for ${tableName}:${key}:`, error);
      throw new Error(`Failed to get KV value: ${error.message}`);
    }
  }

  private async deleteKV(tableName: TABLE_NAMES, key: string): Promise<void> {
    try {
      await this.deleteNamespaceValue(tableName, key);
    } catch (error: any) {
      this.logger.error(`Failed to delete KV value for ${tableName}:${key}:`, error);
      throw new Error(`Failed to delete KV value: ${error.message}`);
    }
  }

  private async listKV(tableName: TABLE_NAMES, options?: ListOptions): Promise<Array<{ name: string }>> {
    try {
      return await this.listNamespaceKeys(tableName, options);
    } catch (error: any) {
      this.logger.error(`Failed to list KV for ${tableName}:`, error);
      throw new Error(`Failed to list KV: ${error.message}`);
    }
  }

  /*---------------------------------------------------------------------------
    Sorted set simulation helpers for message ordering.
    We store an array of objects { id, score } as JSON under a dedicated key.
  ---------------------------------------------------------------------------*/

  private async getSortedMessages(orderKey: string): Promise<Array<{ id: string; score: number }>> {
    const raw = await this.getKV(TABLE_MESSAGES, orderKey);
    if (!raw) return [];
    try {
      const arr = JSON.parse(typeof raw === 'string' ? raw : JSON.stringify(raw));
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      this.logger.error(`Error parsing order data for key ${orderKey}:`, { e });
      return [];
    }
  }

  private async updateSorting(threadMessages: (MastraMessageV1 & { _index?: number })[]) {
    // Sort messages by index or timestamp
    return threadMessages
      .map(msg => ({
        message: msg,
        // Use _index if available, otherwise timestamp, matching Upstash
        score: msg._index !== undefined ? msg._index : msg.createdAt.getTime(),
      }))
      .sort((a, b) => a.score - b.score)
      .map(item => ({
        id: item.message.id,
        score: item.score,
      }));
  }

  private async getIncludedMessagesWithContext(
    threadId: string,
    include: { id: string; withPreviousMessages?: number; withNextMessages?: number }[],
    messageIds: Set<string>,
  ): Promise<void> {
    const threadMessagesKey = this.getThreadMessagesKey(threadId);
    await Promise.all(
      include.map(async item => {
        messageIds.add(item.id);
        if (!item.withPreviousMessages && !item.withNextMessages) return;

        const rank = await this.getRank(threadMessagesKey, item.id);
        if (rank === null) return;

        if (item.withPreviousMessages) {
          const prevIds = await this.getRange(
            threadMessagesKey,
            Math.max(0, rank - item.withPreviousMessages),
            rank - 1,
          );
          prevIds.forEach(id => messageIds.add(id));
        }

        if (item.withNextMessages) {
          const nextIds = await this.getRange(threadMessagesKey, rank + 1, rank + item.withNextMessages);
          nextIds.forEach(id => messageIds.add(id));
        }
      }),
    );
  }

  private async getRecentMessages(threadId: string, limit: number, messageIds: Set<string>): Promise<void> {
    if (limit <= 0) return;

    try {
      const threadMessagesKey = this.getThreadMessagesKey(threadId);
      const latestIds = await this.getLastN(threadMessagesKey, limit);
      latestIds.forEach(id => messageIds.add(id));
    } catch {
      console.log(`No message order found for thread ${threadId}, skipping latest messages`);
    }
  }

  private async fetchAndParseMessages(
    threadId: string,
    messageIds: string[],
  ): Promise<(MastraMessageV1 & { _index?: number })[]> {
    const messages = await Promise.all(
      messageIds.map(async id => {
        try {
          const key = this.getMessageKey(threadId, id);
          const data = await this.getKV(TABLE_MESSAGES, key);
          if (!data) return null;
          return typeof data === 'string' ? JSON.parse(data) : data;
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          this.logger.error(`Error retrieving message ${id}:`, { message });
          return null;
        }
      }),
    );
    return messages.filter((msg): msg is MastraMessageV1 & { _index?: number } => msg !== null);
  }

  /**
   * Queue for serializing sorted order updates.
   * Updates the sorted order for a given key. This operation is eventually consistent.
   */
  private updateQueue = new Map<string, Promise<void>>();

  /**
   * Updates the sorted order for a given key. This operation is eventually consistent.
   * Note: Operations on the same orderKey are serialized using a queue to prevent
   * concurrent updates from conflicting with each other.
   */
  private async updateSortedMessages(
    orderKey: string,
    newEntries: Array<{ id: string; score: number }>,
  ): Promise<void> {
    // Get the current promise chain or create a new one
    const currentPromise = this.updateQueue.get(orderKey) || Promise.resolve();

    // Create the next promise in the chain
    const nextPromise = currentPromise.then(async () => {
      try {
        const currentOrder = await this.getSortedMessages(orderKey);

        // Create a map for faster lookups
        const orderMap = new Map(currentOrder.map(entry => [entry.id, entry]));

        // Update or add new entries
        for (const entry of newEntries) {
          orderMap.set(entry.id, entry);
        }

        // Convert back to array and sort
        const updatedOrder = Array.from(orderMap.values()).sort((a, b) => a.score - b.score);

        // Use putKV for consistent serialization across both APIs
        await this.putKV({
          tableName: TABLE_MESSAGES,
          key: orderKey,
          value: JSON.stringify(updatedOrder),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.logger.error(`Error updating sorted order for key ${orderKey}:`, { message });
        throw error; // Let caller handle the error
      } finally {
        // Clean up the queue if this was the last operation
        if (this.updateQueue.get(orderKey) === nextPromise) {
          this.updateQueue.delete(orderKey);
        }
      }
    });

    // Update the queue with the new promise
    this.updateQueue.set(orderKey, nextPromise);

    // Wait for our turn and handle any errors
    return nextPromise;
  }

  private async getRank(orderKey: string, id: string): Promise<number | null> {
    const order = await this.getSortedMessages(orderKey);
    const index = order.findIndex(item => item.id === id);
    return index >= 0 ? index : null;
  }

  private async getRange(orderKey: string, start: number, end: number): Promise<string[]> {
    const order = await this.getSortedMessages(orderKey);
    const actualStart = start < 0 ? Math.max(0, order.length + start) : start;
    const actualEnd = end < 0 ? order.length + end : Math.min(end, order.length - 1);
    const sliced = order.slice(actualStart, actualEnd + 1);
    return sliced.map(item => item.id);
  }

  private async getLastN(orderKey: string, n: number): Promise<string[]> {
    // Reuse getRange with negative indexing
    return this.getRange(orderKey, -n, -1);
  }

  private async getFullOrder(orderKey: string): Promise<string[]> {
    // Get the full range in ascending order (oldest to newest)
    return this.getRange(orderKey, 0, -1);
  }

  private getKey<T extends TABLE_NAMES>(tableName: T, record: Record<string, string>): string {
    // Add namespace prefix if configured
    const prefix = this.namespacePrefix ? `${this.namespacePrefix}:` : '';
    switch (tableName) {
      case TABLE_THREADS:
        if (!record.id) throw new Error('Thread ID is required');
        return `${prefix}${tableName}:${record.id}`;
      case TABLE_MESSAGES:
        if (!record.threadId || !record.id) throw new Error('Thread ID and Message ID are required');
        return `${prefix}${tableName}:${record.threadId}:${record.id}`;
      case TABLE_WORKFLOW_SNAPSHOT:
        if (!record.namespace || !record.workflow_name || !record.run_id) {
          throw new Error('Namespace, workflow name, and run ID are required');
        }
        let key = `${prefix}${tableName}:${record.namespace}:${record.workflow_name}:${record.run_id}`;
        if (record.resourceId) {
          key = `${key}:${record.resourceId}`;
        }
        return key;
      case TABLE_TRACES:
        if (!record.id) throw new Error('Trace ID is required');
        return `${prefix}${tableName}:${record.id}`;
      default:
        throw new Error(`Unsupported table: ${tableName}`);
    }
  }

  private getSchemaKey(tableName: TABLE_NAMES): string {
    // Add namespace prefix if configured
    const prefix = this.namespacePrefix ? `${this.namespacePrefix}:` : '';
    return `${prefix}schema:${tableName}`;
  }

  private async getTableSchema(tableName: TABLE_NAMES): Promise<Record<string, StorageColumn> | null> {
    try {
      const schemaKey = this.getSchemaKey(tableName);
      return await this.getKV(tableName, schemaKey);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Failed to get schema for ${tableName}:`, { message });
      return null;
    }
  }

  private validateColumnValue(value: unknown, column: StorageColumn): boolean {
    if (value === undefined || value === null) {
      return column.nullable ?? false;
    }

    switch (column.type) {
      case 'text':
      case 'uuid':
        return typeof value === 'string';
      case 'integer':
      case 'bigint':
        return typeof value === 'number';
      case 'timestamp':
        return value instanceof Date || (typeof value === 'string' && !isNaN(Date.parse(value)));
      case 'jsonb':
        if (typeof value !== 'object') return false;
        try {
          JSON.stringify(value);
          return true;
        } catch {
          return false;
        }
      default:
        return false;
    }
  }

  private async validateAgainstSchema(
    record: Record<string, unknown>,
    schema: Record<string, StorageColumn>,
  ): Promise<void> {
    try {
      if (!schema || typeof schema !== 'object' || schema.value === null) {
        throw new Error('Invalid schema format');
      }
      for (const [columnName, column] of Object.entries(schema)) {
        const value = record[columnName];

        // Check primary key presence
        if (column.primaryKey && (value === undefined || value === null)) {
          throw new Error(`Missing primary key value for column ${columnName}`);
        }

        if (!this.validateColumnValue(value, column)) {
          const valueType = value === null ? 'null' : typeof value;
          throw new Error(`Invalid value for column ${columnName}: expected ${column.type}, got ${valueType}`);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error validating record against schema:`, { message, record, schema });
      throw error;
    }
  }

  private async validateRecord<T extends TABLE_NAMES>(record: unknown, tableName: T): Promise<void> {
    try {
      if (!record || typeof record !== 'object') {
        throw new Error('Record must be an object');
      }

      const recordTyped = record as Record<string, unknown>;
      const schema = await this.getTableSchema(tableName);

      // If schema exists, validate against it
      if (schema) {
        await this.validateAgainstSchema(recordTyped, schema);
        return;
      }

      // Fallback validation if no schema found
      switch (tableName) {
        case TABLE_THREADS:
          if (!('id' in recordTyped) || !('resourceId' in recordTyped) || !('title' in recordTyped)) {
            throw new Error('Thread record missing required fields');
          }
          break;
        case TABLE_MESSAGES:
          if (
            !('id' in recordTyped) ||
            !('threadId' in recordTyped) ||
            !('content' in recordTyped) ||
            !('role' in recordTyped)
          ) {
            throw new Error('Message record missing required fields');
          }
          break;
        case TABLE_WORKFLOW_SNAPSHOT:
          if (!('namespace' in recordTyped) || !('workflow_name' in recordTyped) || !('run_id' in recordTyped)) {
            throw new Error('Workflow record missing required fields');
          }
          break;
        case TABLE_TRACES:
          if (!('id' in recordTyped)) {
            throw new Error('Trace record missing required fields');
          }
          break;
        default:
          throw new Error(`Unknown table type: ${tableName}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Failed to validate record for ${tableName}:`, { message, record });
      throw error;
    }
  }

  private ensureMetadata(metadata: Record<string, unknown> | string | undefined): Record<string, unknown> | undefined {
    if (!metadata) return {};
    return typeof metadata === 'string' ? JSON.parse(metadata) : metadata;
  }

  async createTable({
    tableName,
    schema,
  }: {
    tableName: TABLE_NAMES;
    schema: Record<string, StorageColumn>;
  }): Promise<void> {
    try {
      const schemaKey = this.getSchemaKey(tableName);
      const metadata = {
        type: 'table_schema',
        tableName,
        createdAt: new Date().toISOString(),
      };
      await this.putKV({ tableName, key: schemaKey, value: schema, metadata });
    } catch (error: any) {
      this.logger.error(`Failed to store schema for ${tableName}:`, error);
      throw new Error(`Failed to store schema: ${error.message}`);
    }
  }

  /**
   * No-op: This backend is schemaless and does not require schema changes.
   * @param tableName Name of the table
   * @param schema Schema of the table
   * @param ifNotExists Array of column names to add if they don't exist
   */
  async alterTable(_args: {
    tableName: TABLE_NAMES;
    schema: Record<string, StorageColumn>;
    ifNotExists: string[];
  }): Promise<void> {
    // Nothing to do here, Cloudflare KV is schemaless
  }

  async clearTable({ tableName }: { tableName: TABLE_NAMES }): Promise<void> {
    const keys = await this.listKV(tableName);
    if (keys.length > 0) {
      await Promise.all(keys.map(keyObj => this.deleteKV(tableName, keyObj.name)));
    }
  }

  async insert<T extends TABLE_NAMES>({
    tableName,
    record,
  }: {
    tableName: T;
    record: Record<string, any>;
  }): Promise<void> {
    try {
      const key = this.getKey(tableName, record);

      // Process dates and metadata
      const processedRecord = {
        ...record,
        createdAt: record.createdAt ? this.serializeDate(record.createdAt) : undefined,
        updatedAt: record.updatedAt ? this.serializeDate(record.updatedAt) : undefined,
        metadata: record.metadata ? JSON.stringify(record.metadata) : '',
      } as RecordTypes[T];

      // Validate record type
      await this.validateRecord(processedRecord, tableName);
      await this.putKV({ tableName, key, value: processedRecord });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Failed to insert record for ${tableName}:`, { message });
      throw error;
    }
  }

  async load<R>({ tableName, keys }: { tableName: TABLE_NAMES; keys: Record<string, string> }): Promise<R | null> {
    try {
      // Generate key using simplified approach
      const key = this.getKey(tableName, keys as Partial<RecordTypes[typeof tableName]>);

      // Get data from KV store
      const data = await this.getKV(tableName, key);
      if (!data) return null;

      // Handle dates and metadata
      const processed = {
        ...data,
        createdAt: this.ensureDate(data.createdAt),
        updatedAt: this.ensureDate(data.updatedAt),
        metadata: this.ensureMetadata(data.metadata),
      };

      return processed as R;
    } catch (error) {
      this.logger.error(`Failed to load data for ${tableName}:`, {
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  async getThreadById({ threadId }: { threadId: string }): Promise<StorageThreadType | null> {
    const thread = await this.load<StorageThreadType>({ tableName: TABLE_THREADS, keys: { id: threadId } });
    if (!thread) return null;

    try {
      return {
        ...thread,
        createdAt: this.ensureDate(thread.createdAt)!,
        updatedAt: this.ensureDate(thread.updatedAt)!,
        metadata: this.ensureMetadata(thread.metadata),
      };
    } catch (error) {
      this.logger.error(`Error processing thread ${threadId}:`, {
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  async getThreadsByResourceId({ resourceId }: { resourceId: string }): Promise<StorageThreadType[]> {
    try {
      const keyList = await this.listKV(TABLE_THREADS);
      const threads = await Promise.all(
        keyList.map(async keyObj => {
          try {
            const data = await this.getKV(TABLE_THREADS, keyObj.name);
            if (!data) return null;

            const thread = typeof data === 'string' ? JSON.parse(data) : data;
            if (!thread || !thread.resourceId || thread.resourceId !== resourceId) return null;

            return {
              ...thread,
              createdAt: this.ensureDate(thread.createdAt)!,
              updatedAt: this.ensureDate(thread.updatedAt)!,
              metadata: this.ensureMetadata(thread.metadata),
            };
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Error processing thread from key ${keyObj.name}:`, { message });
            return null;
          }
        }),
      );
      return threads.filter((thread): thread is StorageThreadType => thread !== null);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error getting threads for resourceId ${resourceId}:`, { message });
      return [];
    }
  }

  async saveThread({ thread }: { thread: StorageThreadType }): Promise<StorageThreadType> {
    try {
      await this.insert({ tableName: TABLE_THREADS, record: thread });
      return thread;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Error saving thread:', { message });
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
    try {
      const thread = await this.getThreadById({ threadId: id });
      if (!thread) {
        throw new Error(`Thread ${id} not found`);
      }

      const updatedThread = {
        ...thread,
        title,
        metadata: this.ensureMetadata({
          ...(thread.metadata ?? {}),
          ...metadata,
        }),
        updatedAt: new Date(),
      };

      // Insert with proper metadata handling
      await this.insert({ tableName: TABLE_THREADS, record: updatedThread });
      return updatedThread;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error updating thread ${id}:`, { message });
      throw error;
    }
  }

  async deleteThread({ threadId }: { threadId: string }): Promise<void> {
    try {
      // Verify thread exists
      const thread = await this.getThreadById({ threadId });
      if (!thread) {
        throw new Error(`Thread ${threadId} not found`);
      }

      // Get all message keys for this thread first
      const messageKeys = await this.listKV(TABLE_MESSAGES);
      const threadMessageKeys = messageKeys.filter(key => key.name.includes(`${TABLE_MESSAGES}:${threadId}:`));

      // Delete all messages and their order atomically
      await Promise.all([
        // Delete message order
        this.deleteKV(TABLE_MESSAGES, this.getThreadMessagesKey(threadId)),
        // Delete all messages
        ...threadMessageKeys.map(key => this.deleteKV(TABLE_MESSAGES, key.name)),
        // Delete thread
        this.deleteKV(TABLE_THREADS, this.getKey(TABLE_THREADS, { id: threadId })),
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error deleting thread ${threadId}:`, { message });
      throw error;
    }
  }

  private getMessageKey(threadId: string, messageId: string): string {
    try {
      return this.getKey(TABLE_MESSAGES, { threadId, id: messageId });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error getting message key for thread ${threadId} and message ${messageId}:`, { message });
      throw error;
    }
  }
  private getThreadMessagesKey(threadId: string): string {
    try {
      return this.getKey(TABLE_MESSAGES, { threadId, id: 'messages' });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error getting thread messages key for thread ${threadId}:`, { message });
      throw error;
    }
  }

  async saveMessages(args: { messages: MastraMessageV1[]; format?: undefined | 'v1' }): Promise<MastraMessageV1[]>;
  async saveMessages(args: { messages: MastraMessageV2[]; format: 'v2' }): Promise<MastraMessageV2[]>;
  async saveMessages(
    args: { messages: MastraMessageV1[]; format?: undefined | 'v1' } | { messages: MastraMessageV2[]; format: 'v2' },
  ): Promise<MastraMessageV2[] | MastraMessageV1[]> {
    const { messages, format = 'v1' } = args;
    if (!Array.isArray(messages) || messages.length === 0) return [];

    try {
      // Validate message structure and ensure dates
      const validatedMessages = messages.map((message, index) => {
        const errors: string[] = [];
        if (!message.id) errors.push('id is required');
        if (!message.threadId) errors.push('threadId is required');
        if (!message.content) errors.push('content is required');
        if (!message.role) errors.push('role is required');
        if (!message.createdAt) errors.push('createdAt is required');

        if (errors.length > 0) {
          throw new Error(`Invalid message at index ${index}: ${errors.join(', ')}`);
        }

        return {
          ...message,
          createdAt: this.ensureDate(message.createdAt)!,
          type: message.type || 'v2',
          _index: index,
        };
      });

      // Group messages by thread for batch processing
      const messagesByThread = validatedMessages.reduce((acc, message) => {
        if (message.threadId && !acc.has(message.threadId)) {
          acc.set(message.threadId, []);
        }
        if (message.threadId) {
          acc.get(message.threadId)!.push(message as MastraMessageV1 & { _index?: number });
        }
        return acc;
      }, new Map<string, (MastraMessageV1 & { _index?: number })[]>());

      // Process each thread's messages
      await Promise.all(
        Array.from(messagesByThread.entries()).map(async ([threadId, threadMessages]) => {
          try {
            // Verify thread exists
            const thread = await this.getThreadById({ threadId });
            if (!thread) {
              throw new Error(`Thread ${threadId} not found`);
            }

            // Save messages with serialized dates
            await Promise.all(
              threadMessages.map(async message => {
                const key = this.getMessageKey(threadId, message.id);
                // Strip _index and serialize dates before saving
                const { _index, ...cleanMessage } = message;
                const serializedMessage = {
                  ...cleanMessage,
                  createdAt: this.serializeDate(cleanMessage.createdAt),
                };
                await this.putKV({ tableName: TABLE_MESSAGES, key, value: serializedMessage });
              }),
            );

            // Update message order using _index or timestamps
            const orderKey = this.getThreadMessagesKey(threadId);
            const entries = await this.updateSorting(threadMessages);
            await this.updateSortedMessages(orderKey, entries);
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            this.logger.error(`Error processing messages for thread ${threadId}: ${errorMessage}`);
            throw error;
          }
        }),
      );

      // Remove _index from returned messages
      const prepared = validatedMessages.map(
        ({ _index, ...message }) =>
          ({ ...message, type: message.type !== 'v2' ? message.type : undefined }) as MastraMessageV1,
      );
      const list = new MessageList().add(prepared, 'memory');
      if (format === `v2`) return list.get.all.v2();
      return list.get.all.v1();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error saving messages: ${errorMessage}`);
      throw error;
    }
  }

  public async getMessages(args: StorageGetMessagesArg & { format?: 'v1' }): Promise<MastraMessageV1[]>;
  public async getMessages(args: StorageGetMessagesArg & { format: 'v2' }): Promise<MastraMessageV2[]>;
  public async getMessages({
    threadId,
    resourceId,
    selectBy,
    format,
  }: StorageGetMessagesArg & { format?: 'v1' | 'v2' }): Promise<MastraMessageV1[] | MastraMessageV2[]> {
    if (!threadId) throw new Error('threadId is required');

    // Handle selectBy.last type safely - it can be number or false
    let limit = 40; // Default limit
    if (typeof selectBy?.last === 'number') {
      limit = Math.max(0, selectBy.last);
    } else if (selectBy?.last === false) {
      limit = 0;
    }

    const messageIds = new Set<string>();
    if (limit === 0 && !selectBy?.include?.length) return [];

    try {
      // Get included messages and recent messages in parallel
      await Promise.all([
        selectBy?.include?.length
          ? this.getIncludedMessagesWithContext(threadId, selectBy.include, messageIds)
          : Promise.resolve(),
        limit > 0 && !selectBy?.include?.length
          ? this.getRecentMessages(threadId, limit, messageIds)
          : Promise.resolve(),
      ]);

      // Fetch and parse all messages
      const messages = await this.fetchAndParseMessages(threadId, Array.from(messageIds));
      if (!messages.length) return [];

      // Sort messages
      try {
        const threadMessagesKey = this.getThreadMessagesKey(threadId);
        const messageOrder = await this.getFullOrder(threadMessagesKey);
        const orderMap = new Map(messageOrder.map((id, index) => [id, index]));

        messages.sort((a, b) => {
          const indexA = orderMap.get(a.id);
          const indexB = orderMap.get(b.id);

          if (indexA !== undefined && indexB !== undefined) return orderMap.get(a.id)! - orderMap.get(b.id)!;
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.logger.warn(`Error sorting messages, falling back to creation time: ${errorMessage}`);
        messages.sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());
      }

      // Remove _index and ensure dates before returning, just like Upstash
      const prepared = messages.map(({ _index, ...message }) => ({
        ...message,
        type: message.type === (`v2` as `text`) ? undefined : message.type,
        createdAt: this.ensureDate(message.createdAt)!,
      }));
      const list = new MessageList({ threadId, resourceId }).add(prepared as MastraMessageV1[], 'memory');

      if (format === `v1`) return list.get.all.v1();
      return list.get.all.v2();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error retrieving messages for thread ${threadId}: ${errorMessage}`);
      return [];
    }
  }

  private validateWorkflowParams(params: { namespace: string; workflowName: string; runId: string }): void {
    const { namespace, workflowName, runId } = params;
    if (!namespace || !workflowName || !runId) {
      throw new Error('Invalid workflow snapshot parameters');
    }
  }

  private validateWorkflowState(state: any): void {
    if (!state?.runId || !state?.value || !state?.context?.input || !state?.activePaths) {
      throw new Error('Invalid workflow state structure');
    }
  }

  private normalizeSteps(steps: Record<string, any>): Record<
    string,
    {
      status: 'success' | 'waiting' | 'suspended' | 'skipped' | 'failed';
      payload?: any;
      error?: string;
    }
  > {
    const normalizedSteps: Record<
      string,
      {
        status: 'success' | 'waiting' | 'suspended' | 'skipped' | 'failed';
        payload?: any;
        error?: string;
      }
    > = {};

    for (const [stepId, step] of Object.entries(steps)) {
      normalizedSteps[stepId] = {
        status: step.status as 'success' | 'waiting' | 'suspended' | 'skipped' | 'failed',
        payload: step.payload || step.result,
        error: step.error,
      };
    }

    return normalizedSteps;
  }

  private normalizeWorkflowState(data: any): WorkflowRunState {
    return {
      runId: data.runId,
      value: data.value,
      context: data.context,
      serializedStepGraph: data.serializedStepGraph,
      suspendedPaths: data.suspendedPaths || {},
      activePaths: data.activePaths || [],
      timestamp: data.timestamp || Date.now(),
      status: data.status,
      result: data.result,
      error: data.error,
    };
  }

  async persistWorkflowSnapshot(params: {
    namespace: string;
    workflowName: string;
    runId: string;
    snapshot: WorkflowRunState;
  }): Promise<void> {
    try {
      this.validateWorkflowParams(params);
      const { namespace, workflowName, runId, snapshot } = params;

      const normalizedState = this.normalizeWorkflowState(snapshot);
      this.validateWorkflowState(normalizedState);

      await this.insert({
        tableName: TABLE_WORKFLOW_SNAPSHOT,
        record: {
          namespace,
          workflow_name: workflowName,
          run_id: runId,
          snapshot: normalizedState,
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Error persisting workflow snapshot:', { message });
      throw error;
    }
  }

  async loadWorkflowSnapshot(params: {
    namespace: string;
    workflowName: string;
    runId: string;
  }): Promise<WorkflowRunState | null> {
    try {
      this.validateWorkflowParams(params);
      const { namespace, workflowName, runId } = params;

      const key = this.getKey(TABLE_WORKFLOW_SNAPSHOT, { namespace, workflow_name: workflowName, run_id: runId });
      const data = await this.getKV(TABLE_WORKFLOW_SNAPSHOT, key);
      if (!data) return null;

      const state = this.normalizeWorkflowState(data.snapshot || data);
      this.validateWorkflowState(state);
      return state;
    } catch (error) {
      this.logger.error('Error loading workflow snapshot:', {
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  async batchInsert<T extends TABLE_NAMES>(input: { tableName: T; records: Partial<RecordTypes[T]>[] }): Promise<void> {
    if (!input.records || input.records.length === 0) return;

    try {
      await Promise.all(
        input.records.map(async record => {
          // Generate key using simplified approach
          const key = this.getKey(input.tableName, record as Record<string, string>);

          // Process dates and metadata
          const processedRecord = {
            ...record,
            createdAt: record.createdAt ? this.serializeDate(record.createdAt as Date) : undefined,
            updatedAt: record.updatedAt ? this.serializeDate(record.updatedAt as Date) : undefined,
            metadata: record.metadata ? JSON.stringify(record.metadata) : undefined,
          } as RecordTypes[T];

          await this.putKV({ tableName: input.tableName, key, value: processedRecord });
        }),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Error in batch insert:', { message });
      throw error;
    }
  }

  async getTraces({
    name,
    scope,
    page = 0,
    perPage = 100,
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
  }): Promise<any[]> {
    try {
      // Get all keys for traces table
      let keys: string[];
      if (this.bindings) {
        keys = (await this.listKV(TABLE_TRACES))?.map(k => k.name) || [];
      } else {
        const namespaceId = await this.getNamespaceId(TABLE_TRACES);
        const result = await this.client!.kv.namespaces.keys.list(namespaceId, {
          prefix: '',
          limit: 1000,
          account_id: this.accountId!,
        });
        keys = result.result?.map(k => k.name) || [];
      }

      // Fetch all trace records
      const traceRecords = await Promise.all(
        keys.map(async key => {
          const record = await this.getKV(TABLE_TRACES, key);
          if (!record) return null;
          return record;
        }),
      );

      // Filter out nulls and apply filters
      let filteredTraces = traceRecords.filter(
        (record): record is Record<string, any> => record !== null && typeof record === 'object',
      );

      // Apply name filter if provided
      if (name) {
        filteredTraces = filteredTraces.filter(record => record.name?.toLowerCase().startsWith(name.toLowerCase()));
      }

      // Apply scope filter if provided
      if (scope) {
        filteredTraces = filteredTraces.filter(record => record.scope === scope);
      }

      // Apply attribute filters if provided
      if (attributes) {
        filteredTraces = filteredTraces.filter(record => {
          if (!record.attributes) return false;
          const recordAttrs: Record<string, any> | undefined = this.parseJSON(record.attributes);
          if (!recordAttrs) return false;
          return Object.entries(attributes).every(([key, value]) => recordAttrs[key] === value);
        });
      }

      // Apply fromDate filter if provided
      if (fromDate) {
        filteredTraces = filteredTraces.filter(record => new Date(record.createdAt).getTime() >= fromDate.getTime());
      }

      // Apply toDate filter if provided
      if (toDate) {
        filteredTraces = filteredTraces.filter(record => new Date(record.createdAt).getTime() <= toDate.getTime());
      }

      // Sort by createdAt desc
      filteredTraces.sort((a, b) => {
        const dateA = new Date(a.createdAt).getTime();
        const dateB = new Date(b.createdAt).getTime();
        return dateB - dateA;
      });

      // Apply pagination
      const start = page * perPage;
      const end = start + perPage;
      const paginatedTraces = filteredTraces.slice(start, end);

      // Parse JSON fields and return traces
      return paginatedTraces.map(record => ({
        id: record.id,
        parentSpanId: record.parentSpanId,
        traceId: record.traceId,
        name: record.name,
        scope: record.scope,
        kind: record.kind,
        status: this.parseJSON(record.status),
        events: this.parseJSON(record.events) || [],
        links: this.parseJSON(record.links) || [],
        attributes: this.parseJSON(record?.attributes) || {},
        startTime: record.startTime,
        endTime: record.endTime,
        other: this.parseJSON(record.other) || {},
        createdAt: record.createdAt,
      }));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Failed to get traces:', { message });
      return [];
    }
  }

  private parseJSON(value: any): any {
    if (typeof value === 'string') {
      try {
        return JSON.parse(value);
      } catch {
        return value;
      }
    }
    return value;
  }

  getEvalsByAgentName(_agentName: string, _type?: 'test' | 'live'): Promise<EvalRow[]> {
    throw new Error('Method not implemented.');
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

  private buildWorkflowSnapshotPrefix({
    namespace,
    workflowName,
    runId,
    resourceId,
  }: {
    namespace?: string;
    workflowName?: string;
    runId?: string;
    resourceId?: string;
  }): string {
    // Add namespace prefix if configured
    const prefix = this.namespacePrefix ? `${this.namespacePrefix}:` : '';
    let key = `${prefix}${TABLE_WORKFLOW_SNAPSHOT}`;
    if (namespace) key += `:${namespace}`;
    if (workflowName) key += `:${workflowName}`;
    if (runId) key += `:${runId}`;
    if (resourceId) key += `:${resourceId}`;
    // If partial, ensure trailing colon for correct prefix match
    if (!resourceId && (runId || workflowName || namespace)) key += ':';
    return key;
  }

  async getWorkflowRuns({
    namespace,
    workflowName,
    limit = 20,
    offset = 0,
    resourceId,
    fromDate,
    toDate,
  }: {
    namespace?: string;
    workflowName?: string;
    limit?: number;
    offset?: number;
    resourceId?: string;
    fromDate?: Date;
    toDate?: Date;
  } = {}): Promise<WorkflowRuns> {
    try {
      // List all keys in the workflow snapshot table
      const prefix = this.buildWorkflowSnapshotPrefix({ namespace, workflowName });
      const keyObjs = await this.listKV(TABLE_WORKFLOW_SNAPSHOT, { prefix });
      const runs: WorkflowRun[] = [];
      for (const { name: key } of keyObjs) {
        // Extract namespace, workflow_name, run_id, resourceId from key
        const parts = key.split(':');
        const idx = parts.indexOf(TABLE_WORKFLOW_SNAPSHOT);
        if (idx === -1 || parts.length < idx + 4) continue;
        const ns = parts[idx + 1];
        const wfName = parts[idx + 2];
        // If resourceId is present in the key, it's at idx+4
        const keyResourceId = parts.length > idx + 4 ? parts[idx + 4] : undefined;
        // Filter by namespace, workflowName, resourceId if provided
        if ((namespace && ns !== namespace) || (workflowName && wfName !== workflowName)) continue;
        if (resourceId && keyResourceId && keyResourceId !== resourceId) continue;
        // Load the snapshot
        const data = await this.getKV(TABLE_WORKFLOW_SNAPSHOT, key);
        if (!data) continue;
        try {
          // Filter by resourceId in value if not in key
          if (resourceId && data.resourceId && data.resourceId !== resourceId) continue;
          // Filter by fromDate/toDate
          const createdAt = this.ensureDate(data.createdAt);
          if (fromDate && createdAt && createdAt < fromDate) continue;
          if (toDate && createdAt && createdAt > toDate) continue;
          const state = this.normalizeWorkflowState(data.snapshot || data);
          this.validateWorkflowState(state);
          const run = this.parseWorkflowRun({ ...data, snapshot: state });
          runs.push(run);
        } catch (err) {
          this.logger.error('Failed to parse workflow snapshot:', { key, error: err });
        }
      }
      // Sort by createdAt descending
      runs.sort((a, b) => {
        const aDate = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const bDate = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return bDate - aDate;
      });
      // Apply pagination
      const pagedRuns = runs.slice(offset, offset + limit);
      return {
        runs: pagedRuns,
        total: runs.length,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Error in getWorkflowRuns:', { message });
      return { runs: [], total: 0 };
    }
  }

  async getWorkflowRunById({
    namespace,
    runId,
    workflowName,
  }: {
    namespace: string;
    runId: string;
    workflowName: string;
  }): Promise<WorkflowRun | null> {
    try {
      if (!runId || !workflowName || !namespace) {
        throw new Error('runId, workflowName, and namespace are required');
      }
      // Use prefix to list all possible keys for this run (with/without resourceId)
      const prefix = this.buildWorkflowSnapshotPrefix({ namespace, workflowName, runId });
      const keyObjs = await this.listKV(TABLE_WORKFLOW_SNAPSHOT, { prefix });
      if (!keyObjs.length) return null;
      const key = keyObjs[0]?.name;
      const data = await this.getKV(TABLE_WORKFLOW_SNAPSHOT, key as string);
      if (!data) return null;
      // Normalize to WorkflowRun type
      const state = this.normalizeWorkflowState(data.snapshot || data);
      this.validateWorkflowState(state);
      return this.parseWorkflowRun({ ...data, snapshot: state });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.logger.error('Error in getWorkflowRunById:', { message });
      return null;
    }
  }

  async getTracesPaginated(_args: StorageGetTracesArg): Promise<PaginationInfo & { traces: Trace[] }> {
    throw new Error('Method not implemented.');
  }

  async getThreadsByResourceIdPaginated(_args: {
    resourceId: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginationInfo & { threads: StorageThreadType[] }> {
    throw new Error('Method not implemented.');
  }

  async getMessagesPaginated(
    _args: StorageGetMessagesArg,
  ): Promise<PaginationInfo & { messages: MastraMessageV1[] | MastraMessageV2[] }> {
    throw new Error('Method not implemented.');
  }

  async close(): Promise<void> {
    // No explicit cleanup needed
  }
}
