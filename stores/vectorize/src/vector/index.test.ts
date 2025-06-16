import { randomUUID } from 'crypto';
import type { QueryResult } from '@mastra/core';
import dotenv from 'dotenv';
import { describe, it, expect, beforeAll, afterAll, beforeEach, vi, afterEach } from 'vitest';

import { CloudflareVector } from './';

dotenv.config();

vi.setConfig({ testTimeout: 300_000, hookTimeout: 300_000 });

async function waitForIndexDeletion(vector: CloudflareVector, indexName: string) {
  return new Promise((resolve, reject) => {
    const maxAttempts = 30;
    let attempts = 0;
    const interval = setInterval(async () => {
      try {
        const indexes = await vector.listIndexes();
        if (!indexes.includes(indexName)) {
          clearInterval(interval);
          // Add a small delay after confirmed deletion
          setTimeout(resolve, 2000);
          return;
        }
        attempts++;
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          reject(new Error('Timeout waiting for index deletion'));
        }
      } catch (error) {
        // If we get a 404/410, the index is gone
        if (error.status === 404 || error.status === 410) {
          clearInterval(interval);
          // Add a small delay after confirmed deletion
          setTimeout(resolve, 2000);
          return;
        }
        clearInterval(interval);
        reject(error);
      }
    }, 2000);

    setTimeout(() => {
      clearInterval(interval);
      reject(new Error('Global timeout waiting for index deletion'));
    }, 60000);
  });
}

function waitUntilReady(vector: CloudflareVector, indexName: string) {
  return new Promise((resolve, reject) => {
    const maxAttempts = 40;
    let attempts = 0;
    const interval = setInterval(async () => {
      try {
        const stats = await vector.describeIndex({ indexName });
        if (!!stats) {
          clearInterval(interval);
          resolve(true);
        }
        attempts++;
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          reject(new Error('Timeout waiting for index to be ready'));
        }
      } catch (error) {
        // If we get a 410 index deleted, keep waiting
        if (error.status === 410) {
          return;
        }
        clearInterval(interval);
        reject(error);
      }
    }, 2000);

    setTimeout(() => {
      clearInterval(interval);
      reject(new Error('Global timeout waiting for index'));
    }, 80000);
  });
}

function waitUntilVectorsIndexed(
  vector: CloudflareVector,
  indexName: string,
  expectedCount: number,
  exactCount = false,
) {
  return new Promise((resolve, reject) => {
    const maxAttempts = 60; // Increased from 40 to 60 attempts
    let attempts = 0;
    let lastCount = 0;
    let stableCount = 0;

    const interval = setInterval(async () => {
      try {
        const stats = await vector.describeIndex({ indexName });
        const check = exactCount ? stats?.count === expectedCount : stats?.count >= expectedCount;
        if (stats && check) {
          if (stats.count === lastCount) {
            stableCount++;
            if (stableCount >= 3) {
              // Increased stability requirement
              clearInterval(interval);
              resolve(true);
            }
          } else {
            stableCount = 1;
          }
          lastCount = stats.count;
        }
        attempts++;
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          reject(new Error('Timeout waiting for vectors to be indexed'));
        }
      } catch (error) {
        // If we get a 410 index deleted, keep waiting
        if (error.status === 410) {
          return;
        }
        clearInterval(interval);
        reject(error);
      }
    }, 5000); // Increased from 2s to 5s to avoid hammering the API

    setTimeout(() => {
      clearInterval(interval);
      reject(new Error('Global timeout waiting for vectors'));
    }, 240000); // Increased to 4 minutes, less than our 5 minute test timeout
  });
}

function waitForMetadataIndexes(vector: CloudflareVector, indexName: string, expectedCount: number) {
  return new Promise((resolve, reject) => {
    const maxAttempts = 40;
    let attempts = 0;
    const interval = setInterval(async () => {
      try {
        const indexes = await vector.listMetadataIndexes(indexName);
        if (indexes && indexes.length === expectedCount) {
          clearInterval(interval);
          resolve(true);
        }
        attempts++;
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          reject(new Error('Timeout waiting for metadata indexes to be created'));
        }
      } catch (error) {
        clearInterval(interval);
        reject(error);
      }
    }, 2000);

    setTimeout(() => {
      clearInterval(interval);
      reject(new Error('Global timeout waiting for metadata indexes'));
    }, 80000);
  });
}

async function waitForQueryResults({
  vector,
  indexName,
  queryVector,
  expectedCount,
  filter,
  includeVector = false,
}: {
  vector: CloudflareVector;
  indexName: string;
  queryVector: number[];
  expectedCount: number;
  filter?: Record<string, any>;
  includeVector?: boolean;
}): Promise<QueryResult[]> {
  return new Promise((resolve, reject) => {
    const maxAttempts = 40;
    let attempts = 0;
    let lastResults: QueryResult[] = [];
    let stableCount = 0;

    const interval = setInterval(async () => {
      try {
        const results = await vector.query({
          indexName,
          queryVector,
          topK: expectedCount,
          filter,
          includeVector,
        });

        if (results.length === expectedCount) {
          if (JSON.stringify(results) === JSON.stringify(lastResults)) {
            stableCount++;
            if (stableCount >= 2) {
              clearInterval(interval);
              resolve(results);
            }
          } else {
            stableCount = 1;
          }
          lastResults = results;
        }

        attempts++;
        if (attempts >= maxAttempts) {
          clearInterval(interval);
          reject(new Error('Timeout waiting for query results to stabilize'));
        }
      } catch (error) {
        clearInterval(interval);
        reject(error);
      }
    }, 2000);

    setTimeout(() => {
      clearInterval(interval);
      reject(new Error('Global timeout waiting for query results'));
    }, 80000);
  });
}

describe('CloudflareVector', () => {
  let vectorDB: CloudflareVector;
  const VECTOR_DIMENSION = 1536;
  const testIndexName = `default-${randomUUID()}`;
  const testIndexName2 = `default-${randomUUID()}`;

  // Helper function to create a normalized vector
  const createVector = (primaryDimension: number, value: number = 1.0): number[] => {
    const vector = new Array(VECTOR_DIMENSION).fill(0);
    vector[primaryDimension] = value;
    // Normalize the vector for cosine similarity
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(val => val / magnitude);
  };

  beforeAll(() => {
    // Load from environment variables for CI/CD
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    const apiToken = process.env.CLOUDFLARE_API_TOKEN;

    if (!accountId || !apiToken) {
      throw new Error(
        'Missing required environment variables: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, CLOUDFLARE_VECTORIZE_ID',
      );
    }

    vectorDB = new CloudflareVector({ accountId, apiToken });
  });

  afterAll(async () => {
    try {
      await vectorDB.deleteIndex({ indexName: testIndexName });
    } catch (_error) {
      console.warn('Failed to delete test index:', _error);
    }
  });

  describe('Index Operations', () => {
    const tempIndexName = 'test_temp_index';
    const tempIndexNameCreateDescribeDelete = `create-describe-delete-${randomUUID().slice(0, 8)}`;

    beforeEach(async () => {
      // Cleanup any existing index before each test
      try {
        await vectorDB.deleteIndex({ indexName: tempIndexName });
        await waitForIndexDeletion(vectorDB, tempIndexName);
      } catch {
        // Ignore errors if index doesn't exist
      }
    });

    afterEach(async () => {
      // Cleanup after each test
      try {
        await vectorDB.deleteIndex({ indexName: tempIndexName });
        await waitForIndexDeletion(vectorDB, tempIndexName);
      } catch {
        // Ignore errors if index doesn't exist
      }
    });

    afterAll(async () => {
      try {
        // clean up this unique index that only has one test.
        // the test cleans it up but if it fails in the middle it might cause future tests to fail
        await vectorDB.deleteIndex({ indexName: tempIndexNameCreateDescribeDelete });
      } catch {}
    });

    it('should create and list indexes', async () => {
      await vectorDB.createIndex({ indexName: tempIndexName, dimension: VECTOR_DIMENSION, metric: 'cosine' });
      await waitUntilReady(vectorDB, tempIndexName);
      const indexes = await vectorDB.listIndexes();
      expect(indexes).toContain(tempIndexName);
    });

    it('should create, describe, and delete an index', async () => {
      // Create
      await vectorDB.createIndex({
        indexName: tempIndexNameCreateDescribeDelete,
        dimension: VECTOR_DIMENSION,
        metric: 'cosine',
      });
      await waitUntilReady(vectorDB, tempIndexNameCreateDescribeDelete);

      // Describe
      const stats = await vectorDB.describeIndex({ indexName: tempIndexNameCreateDescribeDelete });
      expect(stats).toEqual({
        dimension: VECTOR_DIMENSION,
        metric: 'cosine',
        count: 0,
      });

      try {
        // Delete
        await vectorDB.deleteIndex({ indexName: tempIndexNameCreateDescribeDelete });
      } catch (e) {
        console.error(`Failed deleting index ${tempIndexNameCreateDescribeDelete}`, e);
      }
      const indexes = await vectorDB.listIndexes();
      expect(indexes).not.toContain(tempIndexNameCreateDescribeDelete);
    });
  });

  describe('Vector Operations', () => {
    let vectorIds: string[];
    it('should create index before operations', async () => {
      await vectorDB.createIndex({ indexName: testIndexName, dimension: VECTOR_DIMENSION, metric: 'cosine' });
      await waitUntilReady(vectorDB, testIndexName);
      const indexes = await vectorDB.listIndexes();
      expect(indexes).toContain(testIndexName);
    });

    it('should insert vectors and query them', async () => {
      const testVectors = [createVector(0, 1.0), createVector(1, 1.0), createVector(2, 1.0)];

      const testMetadata = [{ label: 'first-dimension' }, { label: 'second-dimension' }, { label: 'third-dimension' }];

      vectorIds = await vectorDB.upsert({ indexName: testIndexName, vectors: testVectors, metadata: testMetadata });
      expect(vectorIds).toHaveLength(3);

      await waitUntilVectorsIndexed(vectorDB, testIndexName, 3);
      const stats = await vectorDB.describeIndex({ indexName: testIndexName });
      expect(stats.count).toBeGreaterThan(0);

      const results = await waitForQueryResults({
        vector: vectorDB,
        indexName: testIndexName,
        queryVector: createVector(0, 0.9),
        expectedCount: 3,
      });
      expect(results).toHaveLength(3);

      if (results.length > 0) {
        expect(results[0].metadata).toEqual({ label: 'first-dimension' });
      }
    });

    it('should query vectors and return vector in results', async () => {
      await waitUntilVectorsIndexed(vectorDB, testIndexName, 3);

      const queryVector = createVector(0, 0.9);
      const results = await waitForQueryResults({
        vector: vectorDB,
        indexName: testIndexName,
        queryVector,
        expectedCount: 3,
        includeVector: true,
      });

      expect(results).toHaveLength(3);

      for (const result of results) {
        expect(result.vector).toBeDefined();
        expect(result.vector).toHaveLength(VECTOR_DIMENSION);
      }
    });
  });

  describe('Vector update operations', () => {
    const testVectors = [createVector(0, 1.0), createVector(1, 1.0), createVector(2, 1.0)];
    const indexName1 = 'test-index1' + Date.now();
    const indexName2 = 'test-index2' + Date.now();
    const indexName3 = 'test-index3' + Date.now();

    beforeAll(async () => {
      await vectorDB.createIndex({ indexName: indexName1, dimension: VECTOR_DIMENSION, metric: 'cosine' });
      await waitUntilReady(vectorDB, indexName1);
      await vectorDB.createIndex({ indexName: indexName2, dimension: VECTOR_DIMENSION, metric: 'cosine' });
      await waitUntilReady(vectorDB, indexName2);
      await vectorDB.createIndex({ indexName: indexName3, dimension: VECTOR_DIMENSION, metric: 'cosine' });
      await waitUntilReady(vectorDB, indexName3);
    });

    afterAll(async () => {
      try {
        await vectorDB.deleteIndex({ indexName: indexName1 });
      } catch {
        // Ignore errors if index doesn't exist
      }
      try {
        await vectorDB.deleteIndex({ indexName: indexName2 });
      } catch {
        // Ignore errors if index doesn't exist
      }
      try {
        await vectorDB.deleteIndex({ indexName: indexName3 });
      } catch {
        // Ignore errors if index doesn't exist
      }
    });

    it('should update the vector by id', async () => {
      const ids = await vectorDB.upsert({ indexName: indexName1, vectors: testVectors });
      expect(ids).toHaveLength(3);

      const idToBeUpdated = ids[0];
      const newVector = createVector(0, 4.0);
      const newMetaData = {
        test: 'updates',
      };

      const update = {
        vector: newVector,
        metadata: newMetaData,
      };

      await vectorDB.updateVector({ indexName: indexName1, id: idToBeUpdated, update });

      await waitUntilVectorsIndexed(vectorDB, indexName1, 3);

      const results = await waitForQueryResults({
        vector: vectorDB,
        indexName: indexName1,
        queryVector: newVector,
        expectedCount: 3,
        includeVector: true,
      });

      expect(results).toHaveLength(3);
      const updatedResult = results.find(result => result.id === idToBeUpdated);
      expect(updatedResult).toBeDefined();
      expect(updatedResult?.vector).toEqual(newVector);
    });

    it('should only update vector embeddings by id', async () => {
      const ids = await vectorDB.upsert({ indexName: indexName2, vectors: testVectors });
      expect(ids).toHaveLength(3);

      const idToBeUpdated = ids[0];
      const newVector = createVector(0, 4.0);

      const update = {
        vector: newVector,
      };

      await vectorDB.updateVector({ indexName: indexName2, id: idToBeUpdated, update });

      await waitUntilVectorsIndexed(vectorDB, indexName2, 3);

      const results = await waitForQueryResults({
        vector: vectorDB,
        indexName: indexName2,
        queryVector: newVector,
        expectedCount: 2,
        includeVector: true,
      });

      expect(results).toHaveLength(2);
      const updatedResult = results.find(result => result.id === idToBeUpdated);
      expect(updatedResult).toBeDefined();
      expect(updatedResult?.vector).toEqual(newVector);
    });

    it('should throw exception when no updates are given', async () => {
      await expect(vectorDB.updateVector({ indexName: indexName3, id: 'id', update: {} })).rejects.toThrow(
        'No update data provided',
      );
    });
  });

  describe('Vector delete operations', () => {
    const testVectors = [createVector(0, 1.0), createVector(1, 1.0), createVector(2, 1.0)];

    const indexName = 'delete-test-index' + Date.now();

    beforeEach(async () => {
      await vectorDB.createIndex({ indexName, dimension: VECTOR_DIMENSION, metric: 'cosine' });
      await waitUntilReady(vectorDB, indexName);
    });

    afterEach(async () => {
      try {
        await vectorDB.deleteIndex({ indexName });
        await waitForIndexDeletion(vectorDB, indexName);
      } catch {
        // Ignore errors if index doesn't exist
      }
    });

    it('should delete the vector by id', async () => {
      const ids = await vectorDB.upsert({ indexName, vectors: testVectors });
      await waitUntilVectorsIndexed(vectorDB, indexName, testVectors.length);
      expect(ids).toHaveLength(3);
      const idToBeDeleted = ids[0];

      await vectorDB.deleteVector({ indexName, id: idToBeDeleted });
      await waitUntilVectorsIndexed(vectorDB, indexName, 2, true);

      const results = await waitForQueryResults({
        vector: vectorDB,
        indexName,
        queryVector: createVector(0, 1.0),
        expectedCount: 2,
      });

      expect(results).toHaveLength(2);
      expect(results.map(res => res.id)).not.toContain(idToBeDeleted);
    });
  });

  describe('Error Handling', () => {
    it('should handle duplicate index creation gracefully', async () => {
      const duplicateIndexName = `duplicate-test-${randomUUID()}`;
      const dimension = 768;
      const infoSpy = vi.spyOn(vectorDB['logger'], 'info');
      const warnSpy = vi.spyOn(vectorDB['logger'], 'warn');
      try {
        // Create index first time
        await vectorDB.createIndex({
          indexName: duplicateIndexName,
          dimension,
          metric: 'cosine',
        });
        await waitUntilReady(vectorDB, duplicateIndexName);

        // Try to create with same dimensions - should not throw
        await expect(
          vectorDB.createIndex({
            indexName: duplicateIndexName,
            dimension,
            metric: 'cosine',
          }),
        ).resolves.not.toThrow();

        expect(infoSpy).toHaveBeenCalledWith(expect.stringContaining('already exists with'));

        // Try to create with same dimensions and different metric - should not throw
        await expect(
          vectorDB.createIndex({
            indexName: duplicateIndexName,
            dimension,
            metric: 'euclidean',
          }),
        ).resolves.not.toThrow();

        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Attempted to create index with metric'));

        // Try to create with different dimensions - should throw
        await expect(
          vectorDB.createIndex({
            indexName: duplicateIndexName,
            dimension: dimension + 2,
            metric: 'cosine',
          }),
        ).rejects.toThrow(
          `Index "${duplicateIndexName}" already exists with ${dimension} dimensions, but ${dimension + 2} dimensions were requested`,
        );
      } finally {
        infoSpy.mockRestore();
        warnSpy.mockRestore();
        // Cleanup
        await vectorDB.deleteIndex({ indexName: duplicateIndexName });
      }
    });

    it('should handle invalid dimension vectors', async () => {
      await expect(vectorDB.upsert({ indexName: testIndexName, vectors: [[1.0, 0.0]] })).rejects.toThrow();
    });

    it('should handle querying with wrong dimensions', async () => {
      await expect(vectorDB.query({ indexName: testIndexName, queryVector: [1.0, 0.0] })).rejects.toThrow();
    });

    it('should handle non-existent index operations', async () => {
      const nonExistentIndex = 'non_existent_index';
      await expect(
        vectorDB.query({ indexName: nonExistentIndex, queryVector: createVector(0, 1.0) }),
      ).rejects.toThrow();
    });

    it('rejects queries with filter keys longer than 512 characters', async () => {
      const longKey = 'a'.repeat(513);
      const filter = { [longKey]: 'value' };

      await expect(
        vectorDB.query({ indexName: testIndexName, queryVector: createVector(0, 0.9), topK: 10, filter }),
      ).rejects.toThrow();
    });

    it('rejects queries with filter keys containing invalid characters', async () => {
      const invalidFilters = [
        { 'field"name': 'value' }, // Contains "
        { $field: 'value' }, // Contains $
        { '': 'value' }, // Empty key
      ];

      for (const filter of invalidFilters) {
        await expect(
          vectorDB.query({ indexName: testIndexName, queryVector: createVector(0, 0.9), topK: 10, filter }),
        ).rejects.toThrow();
      }
    });

    it('allows queries with valid range operator combinations', async () => {
      const validFilters = [
        { field: { $gt: 5, $lt: 10 } },
        { field: { $gte: 0, $lte: 100 } },
        { field: { $gt: 5, $lte: 10 } },
      ];

      for (const filter of validFilters) {
        await expect(
          vectorDB.query({ indexName: testIndexName, queryVector: createVector(0, 0.9), topK: 10, filter }),
        ).resolves.not.toThrow();
      }
    });

    it('rejects queries with empty object field values', async () => {
      const emptyFilters = { field: {} };
      await expect(
        vectorDB.query({ indexName: testIndexName, queryVector: createVector(0, 0.9), topK: 10, filter: emptyFilters }),
      ).rejects.toThrow();
    });

    it('rejects oversized filter queries', async () => {
      const largeFilter = {
        field1: { $in: Array(1000).fill('test') },
        field2: { $in: Array(1000).fill(123) },
      };

      await expect(
        vectorDB.query({ indexName: testIndexName, queryVector: createVector(0, 0.9), topK: 10, filter: largeFilter }),
      ).rejects.toThrow();
    });

    it('rejects queries with array values in comparison operators', async () => {
      await expect(
        vectorDB.query({
          indexName: testIndexName,
          queryVector: createVector(0, 0.9),
          topK: 10,
          filter: { field: { $gt: [] } },
        }),
      ).rejects.toThrow();

      await expect(
        vectorDB.query({
          indexName: testIndexName,
          queryVector: createVector(0, 0.9),
          topK: 10,
          filter: { field: { $lt: [1, 2, 3] } },
        }),
      ).rejects.toThrow();
    });
  });

  describe('Metadata Filter Tests', () => {
    beforeAll(async () => {
      await vectorDB.createIndex({ indexName: testIndexName2, dimension: VECTOR_DIMENSION, metric: 'cosine' });
      await waitUntilReady(vectorDB, testIndexName2);

      await vectorDB.createMetadataIndex(testIndexName2, 'price', 'number');
      await vectorDB.createMetadataIndex(testIndexName2, 'category', 'string');
      await vectorDB.createMetadataIndex(testIndexName2, 'rating', 'number');
      await vectorDB.createMetadataIndex(testIndexName2, 'nested.number', 'number');
      await vectorDB.createMetadataIndex(testIndexName2, 'nested.string', 'string');
      await vectorDB.createMetadataIndex(testIndexName2, 'nested.boolean', 'boolean');
      await vectorDB.createMetadataIndex(testIndexName2, 'isActive', 'boolean');
      await vectorDB.createMetadataIndex(testIndexName2, 'code', 'string');
      await vectorDB.createMetadataIndex(testIndexName2, 'optionalField', 'string');
      await vectorDB.createMetadataIndex(testIndexName2, 'mixedField', 'string');

      await waitForMetadataIndexes(vectorDB, testIndexName2, 10);

      // Create all test vectors and metadata at once
      const vectors = [
        // Base test vectors
        createVector(0, 1.0),
        createVector(1, 1.0),
        createVector(2, 1.0),
        createVector(3, 1.0),
      ];

      const metadata = [
        // Base test metadata
        {
          price: 100,
          category: 'electronics',
          rating: 4.5,
          nested: {
            number: 100,
            string: 'premium',
            boolean: true,
          },
          isActive: true,
          mixedField: 'string value',
          code: 'A123',
          optionalField: 'exists',
        },
        {
          price: 200,
          category: 'electronics',
          rating: 3.8,
          nested: {
            number: 200,
            string: 'premium',
            boolean: false,
          },
          isActive: false,
          mixedField: 10,
          code: 'B456',
          optionalField: null,
        },
        {
          price: 150,
          category: 'accessories',
          rating: 4.2,
          nested: {
            number: 150,
            string: 'premium',
            boolean: true,
          },
          isActive: false,
          mixedField: false,
          code: 'C789',
        },
        {
          price: 75,
          category: 'accessories',
          rating: 0,
          nested: {
            number: 75,
            string: 'basic',
            boolean: false,
          },
          isActive: false,
          mixedField: true,
        },
      ];

      await vectorDB.upsert({ indexName: testIndexName2, vectors, metadata });
      await waitUntilVectorsIndexed(vectorDB, testIndexName2, vectors.length);

      const stats = await vectorDB.describeIndex({ indexName: testIndexName2 });
      expect(stats.count).toBe(vectors.length);
    });

    afterAll(async () => {
      const currentMetadata = await vectorDB.listMetadataIndexes(testIndexName2);
      for (const { propertyName } of currentMetadata) {
        await vectorDB.deleteMetadataIndex(testIndexName2, propertyName as string);
      }
      try {
        await vectorDB.deleteIndex({ indexName: testIndexName2 });
      } catch {
        // Ignore errors if index doesn't exist
      }
    });

    describe('Basic Equality Operators', () => {
      it('filters with $eq operator', async () => {
        const queryVector = createVector(0, 1.0);
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector,
          expectedCount: 2,
          filter: { category: 'electronics' },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          expect(result.metadata?.category).toBe('electronics');
        });
      });

      it('filters with $ne operator', async () => {
        const queryVector = createVector(0, 1.0);
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector,
          expectedCount: 2,
          filter: { category: { $ne: 'electronics' } },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          expect(result.metadata?.category).not.toBe('electronics');
        });
      });
    });

    describe('Numeric Comparison Operators', () => {
      it('filters with $gt operator', async () => {
        const queryVector = createVector(0, 1.0);
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector,
          expectedCount: 1,
          filter: { price: { $gt: 150 } },
        });
        expect(results.length).toBe(1);
        results.forEach(result => {
          expect(Number(result.metadata?.price)).toBeGreaterThan(150);
        });
      });

      it('filters with $gte operator', async () => {
        const queryVector = createVector(0, 1.0);
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector,
          expectedCount: 3,
          filter: { price: { $gte: 100 } },
        });
        expect(results.length).toBe(3);
        results.forEach(result => {
          const price = Number(result.metadata?.price);
          expect(price).toBeGreaterThanOrEqual(100);
        });
      });

      it('filters with $lt operator', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { price: { $lt: 150 } },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          const price = Number(result.metadata?.price);
          expect(price).toBeLessThan(150);
        });
      });

      it('filters with $lte operator', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 3,
          filter: { price: { $lte: 150 } },
        });
        expect(results.length).toBe(3);
        results.forEach(result => {
          const price = Number(result.metadata?.price);
          expect(price).toBeLessThanOrEqual(150);
        });
      });
    });

    describe('Array Operators', () => {
      it('filters with $in operator for exact matches', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { category: { $in: ['electronics'] } },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          expect(result.metadata?.category).toContain('electronics');
        });
      });

      it('filters with $nin operator', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { category: { $nin: ['electronics'] } },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          expect(result.metadata?.category).not.toContain('electronics');
        });
      });
    });

    describe('Boolean Operations', () => {
      it('filters with boolean values', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { isActive: true },
        });
        expect(results.length).toBe(1);
        expect(results[0]?.metadata?.isActive).toBe(true);
      });

      it('filters with $ne on boolean values', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 3,
          filter: { isActive: { $ne: true } },
        });
        expect(results.length).toBe(3);
        results.forEach(result => {
          expect(result.metadata?.isActive).toBe(false);
        });
      });
    });

    describe('Nested Field Operations', () => {
      it('filters on nested fields with comparison operators', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { 'nested.number': { $gt: 100 } },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          expect(result.metadata?.nested?.number).toBeGreaterThan(100);
        });
      });

      it('combines nested field filters with top-level filters', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { 'nested.number': { $lt: 200 }, category: 'electronics' },
        });
        expect(results.length).toBe(1);
        expect(results[0]?.metadata?.nested?.number).toBeLessThan(200);
        expect(results[0]?.metadata?.category).toBe('electronics');
      });

      it('handles nested string equality', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 3,
          filter: { 'nested.string': 'premium' },
        });
        expect(results.length).toBe(3);
        results.forEach(result => {
          expect(result.metadata?.nested?.string).toBe('premium');
        });
      });

      it('combines nested numeric and boolean conditions', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { 'nested.number': { $gt: 100 }, 'nested.boolean': true },
        });
        expect(results.length).toBe(1);
        expect(results[0]?.metadata?.nested?.number).toBeGreaterThan(100);
        expect(results[0]?.metadata?.nested?.boolean).toBe(true);
      });

      it('handles multiple nested field comparisons', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { 'nested.string': 'premium', 'nested.number': { $lt: 200 }, 'nested.boolean': true },
        });
        expect(results.length).toBe(2);
        const result = results[0]?.metadata?.nested;
        expect(result?.string).toBe('premium');
        expect(result?.number).toBeLessThan(200);
        expect(result?.boolean).toBe(true);
      });

      it('handles $in with nested string values', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 4,
          filter: { 'nested.string': { $in: ['premium', 'basic'] } },
        });
        expect(results.length).toBe(4);
        results.forEach(result => {
          expect(['premium', 'basic']).toContain(result.metadata?.nested?.string);
        });
      });
    });

    describe('String Operations', () => {
      it('handles string numbers in numeric comparisons', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { price: { $gt: '150' } },
        });
        expect(results.length).toBe(1);
        expect(Number(results[0]?.metadata?.price)).toBeGreaterThan(150);
      });

      it('handles mixed numeric and string comparisons', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { price: { $gt: 100 }, category: { $in: ['electronics'] } },
        });
        expect(results.length).toBe(1);
        expect(Number(results[0]?.metadata?.price)).toBeGreaterThan(100);
        expect(results[0]?.metadata?.category).toBe('electronics');
      });
    });

    describe('Filter Validation and Edge Cases', () => {
      it('handles numeric zero values correctly', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { rating: { $eq: 0 } },
        });
        expect(results.length).toBe(1);
        expect(results[0]?.metadata?.rating).toBe(0);
      });

      it('handles multiple conditions on same field', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { price: { $gt: 75, $lt: 200 } },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          const price = Number(result.metadata?.price);
          expect(price).toBeGreaterThan(75);
          expect(price).toBeLessThan(200);
        });
      });

      it('handles exact numeric equality', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { price: { $eq: 100 } },
        });
        expect(results.length).toBe(1);
        expect(results[0]?.metadata?.price).toBe(100);
      });

      it('handles boundary conditions in ranges', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { price: { $gte: 75, $lte: 75 } },
        });
        expect(results.length).toBe(1);
        expect(results[0]?.metadata?.price).toBe(75);
      });
    });

    describe('String Range Queries', () => {
      it('handles lexicographical ordering in string range queries', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { code: { $gt: 'A123', $lt: 'C789' } },
        });
        expect(results.length).toBe(1);
        expect(results[0]?.metadata?.code).toBe('B456');
      });

      it('handles string range queries with special characters', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { code: { $gte: 'A', $lt: 'C' } },
        });
        expect(results.length).toBe(2);
        results.forEach(result => {
          expect(result.metadata?.code).toMatch(/^[AB]/);
        });
      });
    });

    describe('Null and Special Values', () => {
      it('handles $in with null values', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { optionalField: { $in: [null, 'exists'] } },
        });
        expect(results.length).toBe(1);
      });

      it('handles $ne with null values', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 4,
          filter: { optionalField: { $ne: null } },
        });
        expect(results.length).toBe(4);
        expect(results[0]?.metadata?.optionalField).toBe('exists');
      });
    });

    describe('Mixed Type Arrays and Values', () => {
      it('handles $in with mixed type arrays', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 2,
          filter: { mixedField: { $in: ['string value', 10, null] } },
        });
        expect(results.length).toBe(2);
      });

      it('combines different types of filters', async () => {
        const results = await waitForQueryResults({
          vector: vectorDB,
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          expectedCount: 1,
          filter: { mixedField: { $in: ['string value', true] }, price: { $eq: 100 } },
        });
        expect(results.length).toBe(1);
      });
    });

    describe('Filter Size and Structure Validation', () => {
      it('handles filters approaching size limit', async () => {
        // Create a filter that's close to but under 2048 bytes
        const longString = 'a'.repeat(400);
        const filter = {
          category: { $in: [longString, longString.slice(0, 100)] },
          price: { $gt: 0, $lt: 1000 },
          'nested.string': longString.slice(0, 200),
        };

        await expect(
          vectorDB.query({
            indexName: testIndexName2,
            queryVector: createVector(0, 1.0),
            filter,
          }),
        ).resolves.toBeDefined();
      });

      it('handles valid range query combinations', async () => {
        const validRangeCombinations = [
          { price: { $gt: 0, $lt: 1000 } },
          { price: { $gte: 100, $lte: 200 } },
          { price: { $gt: 0, $lte: 1000 } },
          { price: { $gte: 0, $lt: 1000 } },
        ];

        for (const filter of validRangeCombinations) {
          await expect(
            vectorDB.query({
              indexName: testIndexName2,
              queryVector: createVector(0, 1.0),
              filter,
            }),
          ).resolves.toBeDefined();
        }
      });

      it('should handle undefined filter', async () => {
        const results1 = await vectorDB.query({
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          filter: undefined,
        });
        const results2 = await vectorDB.query({
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
        });
        expect(results1).toEqual(results2);
        expect(results1.length).toBeGreaterThan(0);
      });

      it('should handle empty object filter', async () => {
        const results = await vectorDB.query({
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          filter: {},
        });
        const results2 = await vectorDB.query({
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
        });
        expect(results).toEqual(results2);
        expect(results.length).toBeGreaterThan(0);
      });

      it('should handle null filter', async () => {
        const results = await vectorDB.query({
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
          filter: null,
        });
        const results2 = await vectorDB.query({
          indexName: testIndexName2,
          queryVector: createVector(0, 1.0),
        });
        expect(results).toEqual(results2);
        expect(results.length).toBeGreaterThan(0);
      });
    });
  });
});
