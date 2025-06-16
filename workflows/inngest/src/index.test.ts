import { randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import { openai } from '@ai-sdk/openai';
import { serve } from '@hono/node-server';
import { realtimeMiddleware } from '@inngest/realtime';
import { createTool, Mastra, Telemetry } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { RuntimeContext } from '@mastra/core/runtime-context';
import { createHonoServer } from '@mastra/deployer/server';
import { DefaultStorage } from '@mastra/libsql';
import { $ } from 'execa';
import getPort from 'get-port';
import { Inngest } from 'inngest';
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { z } from 'zod';
import { init, serve as inngestServe } from './index';

interface LocalTestContext {
  inngestPort: number;
  handlerPort: number;
  containerName: string;
}

describe('MastraInngestWorkflow', () => {
  beforeEach<LocalTestContext>(async ctx => {
    const inngestPort = await getPort();
    const handlerPort = await getPort();
    const containerName = randomUUID();
    await $`docker run --rm -d --name ${containerName} -p ${inngestPort}:${inngestPort} inngest/inngest:v1.5.10 inngest dev -p ${inngestPort} -u http://host.docker.internal:${handlerPort}/inngest/api`;

    ctx.inngestPort = inngestPort;
    ctx.handlerPort = handlerPort;
    ctx.containerName = containerName;
  });

  afterEach<LocalTestContext>(async ctx => {
    await $`docker stop ${ctx.containerName}`;
  });

  describe('Basic Workflow Execution', () => {
    it('should execute a single step workflow successfully', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);
      const execute = vi.fn<any>().mockResolvedValue({ result: 'success' });
      const step1 = createStep({
        id: 'step1',
        execute,
        inputSchema: z.object({}),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({
          result: z.string(),
        }),
        steps: [step1],
      });
      workflow.then(step1).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(execute).toHaveBeenCalled();
      expect(result.steps['step1']).toEqual({
        status: 'success',
        output: { result: 'success' },
      });

      srv.close();
    });

    it('should execute multiple steps in parallel', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn().mockImplementation(async () => {
        return { value: 'step1' };
      });
      const step2Action = vi.fn().mockImplementation(async () => {
        return { value: 'step2' };
      });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ value: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ value: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({ value: z.string() }),
        steps: [step1, step2],
      });

      workflow.parallel([step1, step2]).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(step1Action).toHaveBeenCalled();
      expect(step2Action).toHaveBeenCalled();
      expect(result.steps).toEqual({
        input: {},
        step1: { status: 'success', output: { value: 'step1' } },
        step2: { status: 'success', output: { value: 'step2' } },
      });

      srv.close();
    });

    it('should execute steps sequentially', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const executionOrder: string[] = [];

      const step1Action = vi.fn().mockImplementation(() => {
        executionOrder.push('step1');
        return { value: 'step1' };
      });
      const step2Action = vi.fn().mockImplementation(() => {
        executionOrder.push('step2');
        return { value: 'step2' };
      });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ value: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({ value: z.string() }),
        outputSchema: z.object({ value: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({ value: z.string() }),
        steps: [step1, step2],
      });

      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(executionOrder).toEqual(['step1', 'step2']);
      expect(result.steps).toEqual({
        input: {},
        step1: { status: 'success', output: { value: 'step1' } },
        step2: { status: 'success', output: { value: 'step2' } },
      });

      srv.close();
    });

    it('should execute a a sleep step', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const execute = vi.fn<any>().mockResolvedValue({ result: 'success' });
      const step1 = createStep({
        id: 'step1',
        execute,
        inputSchema: z.object({}),
        outputSchema: z.object({ result: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: async ({ inputData }) => {
          return { result: 'slept successfully: ' + inputData.result };
        },
        inputSchema: z.object({ result: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({
          result: z.string(),
        }),
        steps: [step1],
      });

      workflow.then(step1).sleep(1000).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const startTime = Date.now();
      const result = await run.start({ inputData: {} });
      const endTime = Date.now();

      expect(execute).toHaveBeenCalled();
      expect(result.steps['step1']).toEqual({
        status: 'success',
        output: { result: 'success' },
        // payload: {},
        // startedAt: expect.any(Number),
        // endedAt: expect.any(Number),
      });

      expect(result.steps['step2']).toEqual({
        status: 'success',
        output: { result: 'slept successfully: success' },
        // payload: { result: 'success' },
        // startedAt: expect.any(Number),
        // endedAt: expect.any(Number),
      });

      expect(endTime - startTime).toBeGreaterThan(1000);

      srv.close();
    });

    it('should execute a a sleep until step', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const execute = vi.fn<any>().mockResolvedValue({ result: 'success' });
      const step1 = createStep({
        id: 'step1',
        execute,
        inputSchema: z.object({}),
        outputSchema: z.object({ result: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: async ({ inputData }) => {
          return { result: 'slept successfully: ' + inputData.result };
        },
        inputSchema: z.object({ result: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({
          result: z.string(),
        }),
        steps: [step1],
      });

      workflow
        .then(step1)
        .sleepUntil(new Date(Date.now() + 1000))
        .then(step2)
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const startTime = Date.now();
      const result = await run.start({ inputData: {} });
      const endTime = Date.now();

      expect(execute).toHaveBeenCalled();
      expect(result.steps['step1']).toEqual({
        status: 'success',
        output: { result: 'success' },
        // payload: {},
        // startedAt: expect.any(Number),
        // endedAt: expect.any(Number),
      });

      expect(result.steps['step2']).toEqual({
        status: 'success',
        output: { result: 'slept successfully: success' },
        // payload: { result: 'success' },
        // startedAt: expect.any(Number),
        // endedAt: expect.any(Number),
      });

      expect(endTime - startTime).toBeGreaterThan(1000);

      srv.close();
    });
  });

  describe('Variable Resolution', () => {
    it('should resolve trigger data', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const execute = vi.fn<any>().mockResolvedValue({ result: 'success' });

      const step1 = createStep({
        id: 'step1',
        execute,
        inputSchema: z.object({ inputData: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute,
        inputSchema: z.object({ result: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({ inputData: z.string() }),
        outputSchema: z.object({}),
      });

      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      const result = await run.start({ inputData: { inputData: 'test-input' } });

      expect(result.steps.step1).toEqual({ status: 'success', output: { result: 'success' } });
      expect(result.steps.step2).toEqual({ status: 'success', output: { result: 'success' } });

      srv.close();
    });

    it('should provide access to step results and trigger data via getStepResult helper', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn().mockImplementation(async ({ inputData }) => {
        // Test accessing trigger data with correct type
        expect(inputData).toEqual({ inputValue: 'test-input' });
        return { value: 'step1-result' };
      });

      const step2Action = vi.fn().mockImplementation(async ({ getStepResult }) => {
        // Test accessing previous step result with type
        const step1Result = getStepResult(step1);
        expect(step1Result).toEqual({ value: 'step1-result' });

        const failedStep = getStepResult(nonExecutedStep);
        expect(failedStep).toBe(null);

        return { value: 'step2-result' };
      });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({ inputValue: z.string() }),
        outputSchema: z.object({ value: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({ value: z.string() }),
        outputSchema: z.object({ value: z.string() }),
      });

      const nonExecutedStep = createStep({
        id: 'non-executed-step',
        execute: vi.fn(),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({ inputValue: z.string() }),
        outputSchema: z.object({ value: z.string() }),
      });

      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      const result = await run.start({ inputData: { inputValue: 'test-input' } });

      expect(step1Action).toHaveBeenCalled();
      expect(step2Action).toHaveBeenCalled();
      expect(result.steps).toEqual({
        input: { inputValue: 'test-input' },
        step1: { status: 'success', output: { value: 'step1-result' } },
        step2: { status: 'success', output: { value: 'step2-result' } },
      });

      srv.close();
    });

    it('should resolve trigger data from context', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const execute = vi.fn<any>().mockResolvedValue({ result: 'success' });
      const triggerSchema = z.object({
        inputData: z.string(),
      });

      const step1 = createStep({
        id: 'step1',
        execute,
        inputSchema: triggerSchema,
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: triggerSchema,
        outputSchema: z.object({ result: z.string() }),
      });

      workflow.then(step1).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      await run.start({ inputData: { inputData: 'test-input' } });

      expect(execute).toHaveBeenCalledWith(
        expect.objectContaining({
          inputData: { inputData: 'test-input' },
        }),
      );

      srv.close();
    });

    it('should resolve trigger data from getInitData', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const execute = vi.fn<any>().mockResolvedValue({ result: 'success' });
      const triggerSchema = z.object({
        cool: z.string(),
      });

      const step1 = createStep({
        id: 'step1',
        execute,
        inputSchema: triggerSchema,
        outputSchema: z.object({ result: z.string() }),
      });

      const step2 = createStep({
        id: 'step2',
        execute: async ({ getInitData }) => {
          const initData = getInitData<typeof workflow>();
          return { result: initData };
        },
        inputSchema: z.object({ result: z.string() }),
        outputSchema: z.object({ result: z.object({ cool: z.string() }) }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: triggerSchema,
        outputSchema: z.object({ result: z.string() }),
      });

      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      const result = await run.start({ inputData: { cool: 'test-input' } });

      expect(execute).toHaveBeenCalledWith(
        expect.objectContaining({
          inputData: { cool: 'test-input' },
        }),
      );

      expect(result.steps.step2).toEqual({ status: 'success', output: { result: { cool: 'test-input' } } });

      srv.close();
    });

    it('should resolve variables from previous steps', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn<any>().mockResolvedValue({
        nested: { value: 'step1-data' },
      });
      const step2Action = vi.fn<any>().mockResolvedValue({ result: 'success' });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ nested: z.object({ value: z.string() }) }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({ previousValue: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({ result: z.string() }),
      });

      workflow
        .then(step1)
        .map({
          previousValue: {
            step: step1,
            path: 'nested.value',
          },
        })
        .then(step2)
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      await run.start({ inputData: {} });

      expect(step2Action).toHaveBeenCalledWith(
        expect.objectContaining({
          inputData: {
            previousValue: 'step1-data',
          },
        }),
      );

      srv.close();
    });
  });

  describe('Simple Conditions', () => {
    it('should follow conditional chains', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn().mockImplementation(() => {
        return Promise.resolve({ status: 'success' });
      });
      const step2Action = vi.fn().mockImplementation(() => {
        return Promise.resolve({ result: 'step2' });
      });
      const step3Action = vi.fn().mockImplementation(() => {
        return Promise.resolve({ result: 'step3' });
      });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({ status: z.string() }),
        outputSchema: z.object({ status: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({ status: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });
      const step3 = createStep({
        id: 'step3',
        execute: step3Action,
        inputSchema: z.object({ status: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({ status: z.string() }),
        outputSchema: z.object({ result: z.string() }),
        steps: [step1, step2, step3],
      });

      workflow
        .then(step1)
        .branch([
          [
            async ({ inputData }) => {
              return inputData.status === 'success';
            },
            step2,
          ],
          [
            async ({ inputData }) => {
              return inputData.status === 'failed';
            },
            step3,
          ],
        ])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      const result = await run.start({ inputData: { status: 'success' } });
      srv.close();

      expect(step1Action).toHaveBeenCalled();
      expect(step2Action).toHaveBeenCalled();
      expect(step3Action).not.toHaveBeenCalled();
      expect(result.steps).toEqual({
        input: { status: 'success' },
        step1: { status: 'success', output: { status: 'success' } },
        step2: { status: 'success', output: { result: 'step2' } },
      });
    });

    it('should handle failing dependencies', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      let err: Error | undefined;
      const step1Action = vi.fn<any>().mockImplementation(() => {
        err = new Error('Failed');
        throw err;
      });
      const step2Action = vi.fn<any>();

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        steps: [step1, step2],
      });

      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      let result: Awaited<ReturnType<typeof run.start>> | undefined = undefined;
      try {
        result = await run.start({ inputData: {} });
      } catch {
        // do nothing
      }

      srv.close();

      expect(step1Action).toHaveBeenCalled();
      expect(step2Action).not.toHaveBeenCalled();
      expect(result?.steps).toEqual({
        input: {},
        step1: { status: 'failed', error: 'Failed' },
      });
    });

    it('should support simple string conditions', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn<any>().mockResolvedValue({ status: 'success' });
      const step2Action = vi.fn<any>().mockResolvedValue({ result: 'step2' });
      const step3Action = vi.fn<any>().mockResolvedValue({ result: 'step3' });
      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ status: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({ status: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });
      const step3 = createStep({
        id: 'step3',
        execute: step3Action,
        inputSchema: z.object({ result: z.string() }),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        steps: [step1, step2, step3],
      });
      workflow
        .then(step1)
        .branch([
          [
            async ({ inputData }) => {
              return inputData.status === 'success';
            },
            step2,
          ],
        ])
        .map({
          result: {
            step: step3,
            path: 'result',
          },
        })
        .branch([
          [
            async ({ inputData }) => {
              return inputData.result === 'unexpected value';
            },
            step3,
          ],
        ])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      const result = await run.start({ inputData: { status: 'success' } });
      srv.close();

      expect(step1Action).toHaveBeenCalled();
      expect(step2Action).toHaveBeenCalled();
      expect(step3Action).not.toHaveBeenCalled();
      expect(result.steps).toMatchObject({
        input: { status: 'success' },
        step1: { status: 'success', output: { status: 'success' } },
        step2: { status: 'success', output: { result: 'step2' } },
      });
    });

    it('should support custom condition functions', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn<any>().mockResolvedValue({ count: 5 });
      const step2Action = vi.fn<any>();

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ count: z.number() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({ count: z.number() }),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      workflow
        .then(step1)
        .branch([
          [
            async ({ getStepResult }) => {
              const step1Result = getStepResult(step1);

              return step1Result ? step1Result.count > 3 : false;
            },
            step2,
          ],
        ])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      const result = await run.start({ inputData: { count: 5 } });
      srv.close();

      expect(step2Action).toHaveBeenCalled();
      expect(result.steps.step1).toEqual({
        status: 'success',
        output: { count: 5 },
      });
      expect(result.steps.step2).toEqual({
        status: 'success',
        output: undefined,
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle step execution errors', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const error = new Error('Step execution failed');
      const failingAction = vi.fn<any>().mockRejectedValue(error);

      const step1 = createStep({
        id: 'step1',
        execute: failingAction,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      workflow.then(step1).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();

      await expect(run.start({ inputData: {} })).resolves.toMatchObject({
        steps: {
          step1: {
            error: 'Step execution failed',
            status: 'failed',
          },
        },
      });

      srv.close();
    });

    it('should handle step execution errors within branches', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const error = new Error('Step execution failed');
      const failingAction = vi.fn<any>().mockRejectedValue(error);
      const successAction = vi.fn<any>().mockResolvedValue({});

      const step1 = createStep({
        id: 'step1',
        execute: successAction,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const step2 = createStep({
        id: 'step2',
        execute: failingAction,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const step3 = createStep({
        id: 'step3',
        execute: successAction,
        inputSchema: z.object({
          step1: z.object({}),
          step2: z.object({}),
        }),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      workflow.parallel([step1, step2]).then(step3).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(result.steps).toMatchObject({
        step1: {
          status: 'success',
        },
        step2: {
          status: 'failed',
          error: 'Step execution failed',
        },
      });

      srv.close();
    });

    it('should handle step execution errors within nested workflows', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const error = new Error('Step execution failed');
      const failingAction = vi.fn<any>().mockRejectedValue(error);
      const successAction = vi.fn<any>().mockResolvedValue({});

      const step1 = createStep({
        id: 'step1',
        execute: successAction,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const step2 = createStep({
        id: 'step2',
        execute: failingAction,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const step3 = createStep({
        id: 'step3',
        execute: successAction,
        inputSchema: z.object({
          step1: z.object({}),
          step2: z.object({}),
        }),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      workflow.parallel([step1, step2]).then(step3).commit();

      const mainWorkflow = createWorkflow({
        id: 'main-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      })
        .then(workflow)
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'main-workflow': mainWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = mainWorkflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(result.steps).toMatchObject({
        'test-workflow': {
          status: 'failed',
          error: 'Step execution failed',
        },
      });

      srv.close();
    });
  });

  describe('Complex Conditions', () => {
    it('should handle nested AND/OR conditions', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn<any>().mockResolvedValue({
        status: 'partial',
        score: 75,
        flags: { isValid: true },
      });
      const step2Action = vi.fn<any>().mockResolvedValue({ result: 'step2' });
      const step3Action = vi.fn<any>().mockResolvedValue({ result: 'step3' });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({
          status: z.string(),
          score: z.number(),
          flags: z.object({ isValid: z.boolean() }),
        }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({
          status: z.string(),
          score: z.number(),
          flags: z.object({ isValid: z.boolean() }),
        }),
        outputSchema: z.object({ result: z.string() }),
      });
      const step3 = createStep({
        id: 'step3',
        execute: step3Action,
        inputSchema: z.object({
          result: z.string(),
        }),
        outputSchema: z.object({ result: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      workflow
        .then(step1)
        .branch([
          [
            async ({ getStepResult }) => {
              const step1Result = getStepResult(step1);
              return (
                step1Result?.status === 'success' || (step1Result?.status === 'partial' && step1Result?.score >= 70)
              );
            },
            step2,
          ],
        ])
        .map({
          result: {
            step: step2,
            path: 'result',
          },
        })
        .branch([
          [
            async ({ inputData, getStepResult }) => {
              const step1Result = getStepResult(step1);
              return !inputData.result || step1Result?.score < 70;
            },
            step3,
          ],
        ])
        .map({
          result: {
            step: step3,
            path: 'result',
          },
        })
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(step2Action).toHaveBeenCalled();
      expect(step3Action).not.toHaveBeenCalled();
      expect(result.steps.step2).toEqual({ status: 'success', output: { result: 'step2' } });

      srv.close();
    });
  });

  describe('Loops', () => {
    it('should run an until loop', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const increment = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)
        const currentValue = inputData.value;

        // Increment the value
        const newValue = currentValue + 1;

        return { value: newValue };
      });
      const incrementStep = createStep({
        id: 'increment',
        description: 'Increments the current value by 1',
        inputSchema: z.object({
          value: z.number(),
          target: z.number(),
        }),
        outputSchema: z.object({
          value: z.number(),
        }),
        execute: increment,
      });

      const final = vi.fn().mockImplementation(async ({ inputData }) => {
        return { finalValue: inputData?.value };
      });
      const finalStep = createStep({
        id: 'final',
        description: 'Final step that prints the result',
        inputSchema: z.object({
          value: z.number(),
        }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: final,
      });

      const counterWorkflow = createWorkflow({
        steps: [incrementStep, finalStep],
        id: 'counter-workflow',
        inputSchema: z.object({
          target: z.number(),
          value: z.number(),
        }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
      });

      counterWorkflow
        .dountil(incrementStep, async ({ inputData }) => {
          return (inputData?.value ?? 0) >= 12;
        })
        .then(finalStep)
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { target: 10, value: 0 } });

      expect(increment).toHaveBeenCalledTimes(12);
      expect(final).toHaveBeenCalledTimes(1);
      // @ts-ignore
      expect(result.result).toEqual({ finalValue: 12 });
      // @ts-ignore
      expect(result.steps.increment.output).toEqual({ value: 12 });

      srv.close();
    });

    it('should run a while loop', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const increment = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)
        const currentValue = inputData.value;

        // Increment the value
        const newValue = currentValue + 1;

        return { value: newValue };
      });
      const incrementStep = createStep({
        id: 'increment',
        description: 'Increments the current value by 1',
        inputSchema: z.object({
          value: z.number(),
          target: z.number(),
        }),
        outputSchema: z.object({
          value: z.number(),
        }),
        execute: increment,
      });

      const final = vi.fn().mockImplementation(async ({ inputData }) => {
        return { finalValue: inputData?.value };
      });
      const finalStep = createStep({
        id: 'final',
        description: 'Final step that prints the result',
        inputSchema: z.object({
          value: z.number(),
        }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: final,
      });

      const counterWorkflow = createWorkflow({
        steps: [incrementStep, finalStep],
        id: 'counter-workflow',
        inputSchema: z.object({
          target: z.number(),
          value: z.number(),
        }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
      });

      counterWorkflow
        .dowhile(incrementStep, async ({ inputData }) => {
          return (inputData?.value ?? 0) < 12;
        })
        .then(finalStep)
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { target: 10, value: 0 } });

      expect(increment).toHaveBeenCalledTimes(12);
      expect(final).toHaveBeenCalledTimes(1);
      // @ts-ignore
      expect(result.result).toEqual({ finalValue: 12 });
      // @ts-ignore
      expect(result.steps.increment.output).toEqual({ value: 12 });

      srv.close();
    });
  });

  describe('foreach', () => {
    it('should run a single item concurrency (default) for loop', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const startTime = Date.now();
      const map = vi.fn().mockImplementation(async ({ inputData }) => {
        await new Promise(resolve => setTimeout(resolve, 1e3));
        return { value: inputData.value + 11 };
      });
      const mapStep = createStep({
        id: 'map',
        description: 'Maps (+11) on the current value',
        inputSchema: z.object({
          value: z.number(),
        }),
        outputSchema: z.object({
          value: z.number(),
        }),
        execute: map,
      });

      const finalStep = createStep({
        id: 'final',
        description: 'Final step that prints the result',
        inputSchema: z.array(z.object({ value: z.number() })),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: async ({ inputData }) => {
          return { finalValue: inputData.reduce((acc, curr) => acc + curr.value, 0) };
        },
      });

      const counterWorkflow = createWorkflow({
        steps: [mapStep, finalStep],
        id: 'counter-workflow',
        inputSchema: z.array(z.object({ value: z.number() })),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
      });

      counterWorkflow.foreach(mapStep).then(finalStep).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: [{ value: 1 }, { value: 22 }, { value: 333 }] });

      const endTime = Date.now();
      const duration = endTime - startTime;
      expect(duration).toBeGreaterThan(1e3 * 3);

      expect(map).toHaveBeenCalledTimes(3);
      expect(result.steps).toEqual({
        input: [{ value: 1 }, { value: 22 }, { value: 333 }],
        map: { status: 'success', output: [{ value: 12 }, { value: 33 }, { value: 344 }] },
        final: { status: 'success', output: { finalValue: 1 + 11 + (22 + 11) + (333 + 11) } },
      });

      srv.close();
    });
  });

  describe('if-else branching', () => {
    it('should run the if-then branch', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const start = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)

        // Increment the value
        const newValue = (inputData?.startValue ?? 0) + 1;

        return { newValue };
      });
      const startStep = createStep({
        id: 'start',
        description: 'Increments the current value by 1',
        inputSchema: z.object({
          startValue: z.number(),
        }),
        outputSchema: z.object({
          newValue: z.number(),
        }),
        execute: start,
      });

      const other = vi.fn().mockImplementation(async () => {
        return { other: 26 };
      });
      const otherStep = createStep({
        id: 'other',
        description: 'Other step',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({
          other: z.number(),
        }),
        execute: other,
      });

      const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
        const startVal = getStepResult(startStep)?.newValue ?? 0;
        const otherVal = getStepResult(otherStep)?.other ?? 0;
        return { finalValue: startVal + otherVal };
      });
      const finalIf = createStep({
        id: 'finalIf',
        description: 'Final step that prints the result',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: final,
      });
      const finalElse = createStep({
        id: 'finalElse',
        description: 'Final step that prints the result',
        inputSchema: z.object({ other: z.number() }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: final,
      });

      const counterWorkflow = createWorkflow({
        id: 'counter-workflow',
        inputSchema: z.object({
          startValue: z.number(),
        }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        steps: [startStep, finalIf],
      });

      const elseBranch = createWorkflow({
        id: 'else-branch',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        steps: [otherStep, finalElse],
      })
        .then(otherStep)
        .then(finalElse)
        .commit();

      counterWorkflow
        .then(startStep)
        .branch([
          [
            async ({ inputData }) => {
              const current = inputData.newValue;
              return !current || current < 5;
            },
            finalIf,
          ],
          [
            async ({ inputData }) => {
              const current = inputData.newValue;
              return current >= 5;
            },
            elseBranch,
          ],
        ])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { startValue: 1 } });

      expect(start).toHaveBeenCalledTimes(1);
      expect(other).toHaveBeenCalledTimes(0);
      expect(final).toHaveBeenCalledTimes(1);
      // @ts-ignore
      expect(result.steps.finalIf.output).toEqual({ finalValue: 2 });
      // @ts-ignore
      expect(result.steps.start.output).toEqual({ newValue: 2 });

      srv.close();
    });

    it('should run the else branch', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const start = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)

        // Increment the value
        const newValue = (inputData?.startValue ?? 0) + 1;

        return { newValue };
      });
      const startStep = createStep({
        id: 'start',
        description: 'Increments the current value by 1',
        inputSchema: z.object({
          startValue: z.number(),
        }),
        outputSchema: z.object({
          newValue: z.number(),
        }),
        execute: start,
      });

      const other = vi.fn().mockImplementation(async ({ inputData }) => {
        return { newValue: inputData.newValue, other: 26 };
      });
      const otherStep = createStep({
        id: 'other',
        description: 'Other step',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({
          other: z.number(),
          newValue: z.number(),
        }),
        execute: other,
      });

      const final = vi.fn().mockImplementation(async ({ inputData }) => {
        const startVal = inputData?.newValue ?? 0;
        const otherVal = inputData?.other ?? 0;
        return { finalValue: startVal + otherVal };
      });
      const finalIf = createStep({
        id: 'finalIf',
        description: 'Final step that prints the result',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: final,
      });
      const finalElse = createStep({
        id: 'finalElse',
        description: 'Final step that prints the result',
        inputSchema: z.object({ other: z.number(), newValue: z.number() }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: final,
      });

      const counterWorkflow = createWorkflow({
        id: 'counter-workflow',
        inputSchema: z.object({
          startValue: z.number(),
        }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        steps: [startStep, finalIf],
      });

      const elseBranch = createWorkflow({
        id: 'else-branch',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        steps: [otherStep, finalElse],
      })
        .then(otherStep)
        .then(finalElse)
        .commit();

      counterWorkflow
        .then(startStep)
        .branch([
          [
            async ({ inputData }) => {
              const current = inputData.newValue;
              return !current || current < 5;
            },
            finalIf,
          ],
          [
            async ({ inputData }) => {
              const current = inputData.newValue;
              return current >= 5;
            },
            elseBranch,
          ],
        ])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { startValue: 6 } });

      expect(start).toHaveBeenCalledTimes(1);
      expect(other).toHaveBeenCalledTimes(1);
      expect(final).toHaveBeenCalledTimes(1);
      // @ts-ignore
      expect(result.steps['else-branch'].output).toEqual({ finalValue: 26 + 6 + 1 });
      // @ts-ignore
      expect(result.steps.start.output).toEqual({ newValue: 7 });

      srv.close();
    });
  });

  describe('Schema Validation', () => {
    it.skip('should validate trigger data against schema', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const triggerSchema = z.object({
        required: z.string(),
        nested: z.object({
          value: z.number(),
        }),
      });

      const step1 = createStep({
        id: 'step1',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success' }),
        inputSchema: z.object({
          required: z.string(),
          nested: z.object({
            value: z.number(),
          }),
        }),
        outputSchema: z.object({
          result: z.string(),
        }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: triggerSchema,
        outputSchema: z.object({}),
        steps: [step1],
      });

      workflow.then(step1).commit();

      // Should fail validation
      await expect(
        workflow.execute({
          inputData: {
            required: 'test',
            // @ts-expect-error
            nested: { value: 'not-a-number' },
          },
        }),
      ).rejects.toThrow();

      // Should pass validation
      const run = workflow.createRun();
      await run.start({
        inputData: {
          required: 'test',
          nested: { value: 42 },
        },
      });
    });
  });

  describe('multiple chains', () => {
    it('should run multiple chains in parallel', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1 = createStep({
        id: 'step1',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success1' }),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step2 = createStep({
        id: 'step2',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success2' }),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step3 = createStep({
        id: 'step3',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success3' }),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step4 = createStep({
        id: 'step4',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success4' }),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step5 = createStep({
        id: 'step5',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success5' }),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        steps: [step1, step2, step3, step4, step5],
      });
      workflow
        .parallel([
          createWorkflow({
            id: 'nested-a',
            inputSchema: z.object({}),
            outputSchema: z.object({}),
            steps: [step1, step2, step3],
          })
            .then(step1)
            .then(step2)
            .then(step3)
            .commit(),
          createWorkflow({
            id: 'nested-b',
            inputSchema: z.object({}),
            outputSchema: z.object({}),
            steps: [step4, step5],
          })
            .then(step4)
            .then(step5)
            .commit(),
        ])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(result.steps['nested-a']).toEqual({ status: 'success', output: { result: 'success3' } });
      expect(result.steps['nested-b']).toEqual({ status: 'success', output: { result: 'success5' } });

      srv.close();
    });
  });

  describe('Retry', () => {
    it('should retry a step default 0 times', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1 = createStep({
        id: 'step1',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success' }),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step2 = createStep({
        id: 'step2',
        execute: vi.fn<any>().mockRejectedValue(new Error('Step failed')),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(result.steps.step1).toEqual({ status: 'success', output: { result: 'success' } });
      expect(result.steps.step2).toEqual({ status: 'failed', error: 'Step failed' });
      expect(step1.execute).toHaveBeenCalledTimes(1);
      expect(step2.execute).toHaveBeenCalledTimes(1); // 0 retries + 1 initial call

      srv.close();
    });

    // Need to fix so we can throw for inngest to recognize retries
    it.skip('should retry a step with a custom retry config', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1 = createStep({
        id: 'step1',
        execute: vi.fn<any>().mockResolvedValue({ result: 'success' }),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step2 = createStep({
        id: 'step2',
        execute: vi.fn<any>().mockRejectedValue(new Error('Step failed')),
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        retryConfig: { attempts: 5, delay: 200 },
      });

      new Mastra({
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      workflow.then(step1).then(step2).commit();

      const run = workflow.createRun();
      const result = await run.start({ inputData: {} });

      expect(result.steps.step1).toEqual({ status: 'success', output: { result: 'success' } });
      expect(result.steps.step2).toEqual({ status: 'failed', error: 'Step failed' });
      expect(step1.execute).toHaveBeenCalledTimes(1);
      expect(step2.execute).toHaveBeenCalledTimes(6); // 5 retries + 1 initial call
    });
  });

  describe('Interoperability (Actions)', () => {
    it('should be able to use all action types in a workflow', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn<any>().mockResolvedValue({ name: 'step1' });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ name: z.string() }),
      });

      // @ts-ignore
      const toolAction = vi.fn<any>().mockImplementation(async ({ context }) => {
        return { name: context.name };
      });

      const randomTool = createTool({
        id: 'random-tool',
        execute: toolAction,
        description: 'random-tool',
        inputSchema: z.object({ name: z.string() }),
        outputSchema: z.object({ name: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({ name: z.string() }),
      });

      workflow.then(step1).then(createStep(randomTool)).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const result = await workflow.createRun().start({ inputData: {} });

      srv.close();

      expect(step1Action).toHaveBeenCalled();
      expect(toolAction).toHaveBeenCalled();
      expect(result.steps.step1).toEqual({ status: 'success', output: { name: 'step1' } });
      expect(result.steps['random-tool']).toEqual({ status: 'success', output: { name: 'step1' } });
    }, 10000);
  });

  describe('Watch', () => {
    it('should watch workflow state changes and call onTransition', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        middleware: [realtimeMiddleware()],
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn<any>().mockResolvedValue({ result: 'success1' });
      const step2Action = vi.fn<any>().mockResolvedValue({ result: 'success2' });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({ value: z.string() }),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({ value: z.string() }),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        steps: [step1, step2],
      });
      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();

      // Start watching the workflow
      let cnt = 0;
      let resps: any[] = [];
      run.watch(d => {
        cnt++;
        resps.push(d);
      });

      const executionResult = await run.start({ inputData: {} });
      await new Promise(resolve => setTimeout(resolve, 2000));

      expect(cnt).toBe(5);
      expect(resps.length).toBe(5);
      expect(resps[0]).toMatchObject({
        type: 'watch',
        payload: {
          currentStep: {
            id: 'step1',
            status: 'running',
          },
          workflowState: {
            status: 'running',
            result: null,
            error: null,
            steps: {
              step1: {
                status: 'running',
              },
            },
          },
        },
      });
      expect(resps[1]).toMatchObject({
        type: 'watch',
        payload: {
          currentStep: {
            id: 'step1',
            status: 'success',
            output: { result: 'success1' },
          },
          workflowState: {
            status: 'running',
            result: null,
            error: null,
            steps: {},
          },
        },
      });
      expect(resps[2]).toMatchObject({
        type: 'watch',
        payload: {
          currentStep: {
            id: 'step2',
            status: 'running',
          },
          workflowState: {
            status: 'running',
            result: null,
            error: null,
            steps: {
              step1: {
                status: 'success',
                output: { result: 'success1' },
              },
              step2: {
                status: 'running',
              },
            },
          },
        },
      });

      expect(resps[3]).toMatchObject({
        type: 'watch',
        payload: {
          currentStep: {
            id: 'step2',
            status: 'success',
            output: { result: 'success2' },
          },
          workflowState: {
            status: 'running',
            result: null,
            error: null,
            steps: {},
          },
        },
      });

      expect(resps[resps.length - 1].currentStep).toBeUndefined();
      expect(resps[resps.length - 1]).toMatchObject({
        type: 'watch',
        payload: {
          workflowState: {
            status: 'success',
            result: {
              result: 'success2',
            },
            steps: {
              step1: {
                status: 'success',
                output: { result: 'success1' },
              },
              step2: {
                status: 'success',
                output: { result: 'success2' },
              },
            },
          },
        },
      });

      // Verify execution completed successfully
      expect(executionResult.steps.step1).toEqual({
        status: 'success',
        output: { result: 'success1' },
      });
      expect(executionResult.steps.step2).toEqual({
        status: 'success',
        output: { result: 'success2' },
      });

      srv.close();
    });

    it('should unsubscribe from transitions when unwatch is called', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        middleware: [realtimeMiddleware()],
      });

      const { createWorkflow, createStep } = init(inngest);

      const step1Action = vi.fn<any>().mockResolvedValue({ result: 'success1' });
      const step2Action = vi.fn<any>().mockResolvedValue({ result: 'success2' });

      const step1 = createStep({
        id: 'step1',
        execute: step1Action,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const step2 = createStep({
        id: 'step2',
        execute: step2Action,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        steps: [step1, step2],
      });
      workflow.then(step1).then(step2).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const onTransition = vi.fn();
      const onTransition2 = vi.fn();

      const run = workflow.createRun();

      run.watch(onTransition);
      run.watch(onTransition2);

      await run.start({ inputData: {} });

      expect(onTransition).toHaveBeenCalledTimes(5);
      expect(onTransition2).toHaveBeenCalledTimes(5);

      const run2 = workflow.createRun();

      run2.watch(onTransition2);

      await run2.start({ inputData: {} });

      expect(onTransition).toHaveBeenCalledTimes(5);
      expect(onTransition2).toHaveBeenCalledTimes(10);

      const run3 = workflow.createRun();

      run3.watch(onTransition);

      await run3.start({ inputData: {} });

      srv.close();

      expect(onTransition).toHaveBeenCalledTimes(10);
      expect(onTransition2).toHaveBeenCalledTimes(10);
    });
  });

  describe('Suspend and Resume', () => {
    afterAll(async () => {
      const pathToDb = path.join(process.cwd(), 'mastra.db');

      if (fs.existsSync(pathToDb)) {
        fs.rmSync(pathToDb);
      }
    });
    it('should return the correct runId', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow } = init(inngest);

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        steps: [],
      });
      const run = workflow.createRun();
      const run2 = workflow.createRun({ runId: run.runId });

      expect(run.runId).toBeDefined();
      expect(run2.runId).toBeDefined();
      expect(run.runId).toBe(run2.runId);
    });

    it('should handle basic suspend and resume flow', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        middleware: [realtimeMiddleware()],
      });

      const { createWorkflow, createStep } = init(inngest);

      const getUserInputAction = vi.fn().mockResolvedValue({ userInput: 'test input' });
      const promptAgentAction = vi
        .fn()
        .mockImplementationOnce(async ({ suspend }) => {
          await suspend();
          return undefined;
        })
        .mockImplementationOnce(() => ({ modelOutput: 'test output' }));
      const evaluateToneAction = vi.fn().mockResolvedValue({
        toneScore: { score: 0.8 },
        completenessScore: { score: 0.7 },
      });
      const improveResponseAction = vi.fn().mockResolvedValue({ improvedOutput: 'improved output' });
      const evaluateImprovedAction = vi.fn().mockResolvedValue({
        toneScore: { score: 0.9 },
        completenessScore: { score: 0.8 },
      });

      const getUserInput = createStep({
        id: 'getUserInput',
        execute: getUserInputAction,
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({ userInput: z.string() }),
      });
      const promptAgent = createStep({
        id: 'promptAgent',
        execute: promptAgentAction,
        inputSchema: z.object({ userInput: z.string() }),
        outputSchema: z.object({ modelOutput: z.string() }),
      });
      const evaluateTone = createStep({
        id: 'evaluateToneConsistency',
        execute: evaluateToneAction,
        inputSchema: z.object({ modelOutput: z.string() }),
        outputSchema: z.object({
          toneScore: z.any(),
          completenessScore: z.any(),
        }),
      });
      const improveResponse = createStep({
        id: 'improveResponse',
        execute: improveResponseAction,
        inputSchema: z.object({ toneScore: z.any(), completenessScore: z.any() }),
        outputSchema: z.object({ improvedOutput: z.string() }),
      });
      const evaluateImproved = createStep({
        id: 'evaluateImprovedResponse',
        execute: evaluateImprovedAction,
        inputSchema: z.object({ improvedOutput: z.string() }),
        outputSchema: z.object({
          toneScore: z.any(),
          completenessScore: z.any(),
        }),
      });

      const promptEvalWorkflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({}),
        steps: [getUserInput, promptAgent, evaluateTone, improveResponse, evaluateImproved],
      });

      promptEvalWorkflow
        .then(getUserInput)
        .then(promptAgent)
        .then(evaluateTone)
        .then(improveResponse)
        .then(evaluateImproved)
        .commit();

      // Create a new storage instance for initial run
      const initialStorage = new DefaultStorage({
        url: 'file::memory:',
      });
      const mastra = new Mastra({
        storage: initialStorage,
        workflows: {
          'test-workflow': promptEvalWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = promptEvalWorkflow.createRun();

      // Create a promise to track when the workflow is ready to resume
      let resolveWorkflowSuspended: (value: unknown) => void;
      const workflowSuspended = new Promise(resolve => {
        resolveWorkflowSuspended = resolve;
      });

      run.watch(data => {
        const isPromptAgentSuspended =
          data?.payload?.currentStep?.id === 'promptAgent' && data?.payload?.currentStep?.status === 'suspended';
        if (isPromptAgentSuspended) {
          resolveWorkflowSuspended({ stepId: 'promptAgent', context: { userInput: 'test input for resumption' } });
        }
      });

      const initialResult = await run.start({ inputData: { input: 'test' } });
      expect(initialResult.steps.promptAgent.status).toBe('suspended');
      expect(promptAgentAction).toHaveBeenCalledTimes(1);

      // Wait for the workflow to be ready to resume
      const resumeData = await workflowSuspended;
      const resumeResult = await run.resume({ resumeData: resumeData as any, step: promptAgent });

      srv.close();

      if (!resumeResult) {
        throw new Error('Resume failed to return a result');
      }

      expect(resumeResult.steps).toEqual({
        input: { input: 'test' },
        getUserInput: { status: 'success', output: { userInput: 'test input' } },
        promptAgent: { status: 'success', output: { modelOutput: 'test output' } },
        evaluateToneConsistency: {
          status: 'success',
          output: { toneScore: { score: 0.8 }, completenessScore: { score: 0.7 } },
        },
        improveResponse: { status: 'success', output: { improvedOutput: 'improved output' } },
        evaluateImprovedResponse: {
          status: 'success',
          output: { toneScore: { score: 0.9 }, completenessScore: { score: 0.8 } },
        },
      });
    });

    it('should handle parallel steps with conditional suspend', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        middleware: [realtimeMiddleware()],
      });

      const { createWorkflow, createStep } = init(inngest);

      const getUserInputAction = vi.fn().mockResolvedValue({ userInput: 'test input' });
      const promptAgentAction = vi.fn().mockResolvedValue({ modelOutput: 'test output' });
      const evaluateToneAction = vi.fn().mockResolvedValue({
        toneScore: { score: 0.8 },
        completenessScore: { score: 0.7 },
      });
      const humanInterventionAction = vi
        .fn()
        .mockImplementationOnce(async ({ suspend, resumeData }) => {
          if (!resumeData?.humanPrompt) {
            await suspend();
          }
        })
        .mockImplementationOnce(() => ({ improvedOutput: 'human intervention output' }));
      const explainResponseAction = vi.fn().mockResolvedValue({
        improvedOutput: 'explanation output',
      });

      const getUserInput = createStep({
        id: 'getUserInput',
        execute: getUserInputAction,
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({ userInput: z.string() }),
      });
      const promptAgent = createStep({
        id: 'promptAgent',
        execute: promptAgentAction,
        inputSchema: z.object({ userInput: z.string() }),
        outputSchema: z.object({ modelOutput: z.string() }),
      });
      const evaluateTone = createStep({
        id: 'evaluateToneConsistency',
        execute: evaluateToneAction,
        inputSchema: z.object({ modelOutput: z.string() }),
        outputSchema: z.object({
          toneScore: z.any(),
          completenessScore: z.any(),
        }),
      });
      const humanIntervention = createStep({
        id: 'humanIntervention',
        execute: humanInterventionAction,
        inputSchema: z.object({ toneScore: z.any(), completenessScore: z.any() }),
        outputSchema: z.object({ improvedOutput: z.string() }),
      });
      const explainResponse = createStep({
        id: 'explainResponse',
        execute: explainResponseAction,
        inputSchema: z.object({ toneScore: z.any(), completenessScore: z.any() }),
        outputSchema: z.object({ improvedOutput: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({}),
        steps: [getUserInput, promptAgent, evaluateTone, humanIntervention, explainResponse],
      });

      workflow
        .then(getUserInput)
        .then(promptAgent)
        .then(evaluateTone)
        .branch([
          [() => Promise.resolve(true), humanIntervention],
          [() => Promise.resolve(false), explainResponse],
        ])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();

      const started = run.start({ inputData: { input: 'test' } });

      const result = await new Promise<any>((resolve, reject) => {
        let hasResumed = false;
        run.watch(async data => {
          const suspended =
            data.payload?.currentStep?.id === 'humanIntervention' && data.payload?.currentStep?.status === 'suspended';
          if (suspended) {
            if (!hasResumed) {
              hasResumed = true;

              try {
                const resumed = await run.resume({
                  step: humanIntervention,
                  resumeData: {
                    humanPrompt: 'What improvements would you suggest?',
                  },
                });

                resolve(resumed as any);
              } catch (error) {
                reject(error);
              }
            }
          }
        });
      });

      const initialResult = await started;

      expect(initialResult.steps.humanIntervention.status).toBe('suspended');
      expect(initialResult.steps.explainResponse).toBeUndefined();
      expect(humanInterventionAction).toHaveBeenCalledTimes(2);
      expect(explainResponseAction).not.toHaveBeenCalled();

      srv.close();

      if (!result) {
        throw new Error('Resume failed to return a result');
      }

      expect(result.steps).toEqual({
        input: { input: 'test' },
        getUserInput: { status: 'success', output: { userInput: 'test input' } },
        promptAgent: { status: 'success', output: { modelOutput: 'test output' } },
        evaluateToneConsistency: {
          status: 'success',
          output: { toneScore: { score: 0.8 }, completenessScore: { score: 0.7 } },
        },
        humanIntervention: { status: 'success', output: { improvedOutput: 'human intervention output' } },
      });
    });

    it('should handle complex workflow with multiple suspends', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        middleware: [realtimeMiddleware()],
      });

      const { createWorkflow, createStep } = init(inngest);

      const getUserInputAction = vi.fn().mockResolvedValue({ userInput: 'test input' });
      const promptAgentAction = vi.fn().mockResolvedValue({ modelOutput: 'test output' });

      const evaluateToneAction = vi.fn().mockResolvedValue({
        toneScore: { score: 0.8 },
        completenessScore: { score: 0.7 },
      });
      const improveResponseAction = vi
        .fn()
        .mockImplementationOnce(async ({ suspend }) => {
          await suspend();
        })
        .mockImplementationOnce(() => ({ improvedOutput: 'improved output' }));
      const evaluateImprovedAction = vi.fn().mockResolvedValue({
        toneScore: { score: 0.9 },
        completenessScore: { score: 0.8 },
      });
      const humanInterventionAction = vi
        .fn()
        .mockImplementationOnce(async ({ suspend }) => {
          await suspend();
        })
        .mockImplementationOnce(() => {
          return { improvedOutput: 'human intervention output' };
        });
      const explainResponseAction = vi.fn().mockResolvedValue({
        improvedOutput: 'explanation output',
      });

      const getUserInput = createStep({
        id: 'getUserInput',
        execute: getUserInputAction,
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({ userInput: z.string() }),
      });
      const promptAgent = createStep({
        id: 'promptAgent',
        execute: promptAgentAction,
        inputSchema: z.object({ userInput: z.string() }),
        outputSchema: z.object({ modelOutput: z.string() }),
      });
      const evaluateTone = createStep({
        id: 'evaluateToneConsistency',
        execute: evaluateToneAction,
        inputSchema: z.object({ modelOutput: z.string() }),
        outputSchema: z.object({
          toneScore: z.any(),
          completenessScore: z.any(),
        }),
      });
      const improveResponse = createStep({
        id: 'improveResponse',
        execute: improveResponseAction,
        inputSchema: z.object({ toneScore: z.any(), completenessScore: z.any() }),
        outputSchema: z.object({ improvedOutput: z.string() }),
      });
      const evaluateImproved = createStep({
        id: 'evaluateImprovedResponse',
        execute: evaluateImprovedAction,
        inputSchema: z.object({ improvedOutput: z.string() }),
        outputSchema: z.object({
          toneScore: z.any(),
          completenessScore: z.any(),
        }),
      });
      const humanIntervention = createStep({
        id: 'humanIntervention',
        execute: humanInterventionAction,
        inputSchema: z.object({ toneScore: z.any(), completenessScore: z.any() }),
        outputSchema: z.object({ improvedOutput: z.string() }),
      });
      const explainResponse = createStep({
        id: 'explainResponse',
        execute: explainResponseAction,
        inputSchema: z.object({ toneScore: z.any(), completenessScore: z.any() }),
        outputSchema: z.object({ improvedOutput: z.string() }),
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({}),
        steps: [
          getUserInput,
          promptAgent,
          evaluateTone,
          improveResponse,
          evaluateImproved,
          humanIntervention,
          explainResponse,
        ],
      });

      workflow
        .then(getUserInput)
        .then(promptAgent)
        .then(evaluateTone)
        .then(improveResponse)
        .then(evaluateImproved)
        .map({
          toneScore: {
            step: evaluateTone,
            path: 'toneScore',
          },
          completenessScore: {
            step: evaluateTone,
            path: 'completenessScore',
          },
        })
        .parallel([humanIntervention, explainResponse])
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const started = run.start({ inputData: { input: 'test' } });
      let improvedResponseResultPromise: Promise<any | undefined>;

      const resultPromise = new Promise<any>((resolve, reject) => {
        let hasResumed = false;
        let hasResumedImproveResponse = false;
        run.watch(async data => {
          const state = data.payload?.workflowState;

          if (state.status !== 'suspended') {
            return;
          }

          const isHumanInterventionSuspended = state.steps?.humanIntervention?.status === 'suspended';
          const isImproveResponseSuspended = state.steps?.improveResponse?.status === 'suspended';

          if (isHumanInterventionSuspended) {
            if (!hasResumed) {
              hasResumed = true;

              try {
                const resumed = await run.resume({
                  step: humanIntervention,
                  resumeData: {
                    humanPrompt: 'What improvements would you suggest?',
                  },
                });
                resolve(resumed as any);
              } catch (error) {
                reject(error);
              }
            }
          } else if (isImproveResponseSuspended) {
            if (!hasResumedImproveResponse) {
              hasResumedImproveResponse = true;
              const resumed = run.resume({
                step: improveResponse,
              });
              improvedResponseResultPromise = resumed;
            }
          }
        });
      });

      const result = await resultPromise;
      const initialResult = await started;
      expect(initialResult?.steps.improveResponse.status).toBe('suspended');
      // @ts-ignore
      const improvedResponseResult = await improvedResponseResultPromise;

      expect(improvedResponseResult?.steps.humanIntervention.status).toBe('suspended');
      expect(improvedResponseResult?.steps.improveResponse.status).toBe('success');
      expect(improvedResponseResult?.steps.evaluateImprovedResponse.status).toBe('success');

      srv.close();
      if (!result) {
        throw new Error('Resume failed to return a result');
      }

      expect(humanInterventionAction).toHaveBeenCalledTimes(2);
      expect(explainResponseAction).toHaveBeenCalledTimes(1);

      expect(result.steps).toMatchObject({
        input: { input: 'test' },
        getUserInput: { status: 'success', output: { userInput: 'test input' } },
        promptAgent: { status: 'success', output: { modelOutput: 'test output' } },
        evaluateToneConsistency: {
          status: 'success',
          output: { toneScore: { score: 0.8 }, completenessScore: { score: 0.7 } },
        },
        improveResponse: { status: 'success', output: { improvedOutput: 'improved output' } },
        evaluateImprovedResponse: {
          status: 'success',
          output: { toneScore: { score: 0.9 }, completenessScore: { score: 0.8 } },
        },
        humanIntervention: { status: 'success', output: { improvedOutput: 'human intervention output' } },
      });
    });

    it('should handle basic suspend and resume flow with async await syntax', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);
      const getUserInputAction = vi.fn().mockResolvedValue({ userInput: 'test input' });
      const promptAgentAction = vi
        .fn()
        .mockImplementationOnce(async ({ suspend }) => {
          await suspend({ testPayload: 'hello' });
          return undefined;
        })
        .mockImplementationOnce(() => ({ modelOutput: 'test output' }));
      const evaluateToneAction = vi.fn().mockResolvedValue({
        toneScore: { score: 0.8 },
        completenessScore: { score: 0.7 },
      });
      const improveResponseAction = vi
        .fn()
        .mockImplementationOnce(async ({ suspend }) => {
          await suspend();
          return undefined;
        })
        .mockImplementationOnce(() => ({ improvedOutput: 'improved output' }));
      const evaluateImprovedAction = vi.fn().mockResolvedValue({
        toneScore: { score: 0.9 },
        completenessScore: { score: 0.8 },
      });

      const getUserInput = createStep({
        id: 'getUserInput',
        execute: getUserInputAction,
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({ userInput: z.string() }),
      });
      const promptAgent = createStep({
        id: 'promptAgent',
        execute: promptAgentAction,
        inputSchema: z.object({ userInput: z.string() }),
        outputSchema: z.object({ modelOutput: z.string() }),
        suspendSchema: z.object({ testPayload: z.string() }),
        resumeSchema: z.object({ userInput: z.string() }),
      });
      const evaluateTone = createStep({
        id: 'evaluateToneConsistency',
        execute: evaluateToneAction,
        inputSchema: z.object({ modelOutput: z.string() }),
        outputSchema: z.object({
          toneScore: z.any(),
          completenessScore: z.any(),
        }),
      });
      const improveResponse = createStep({
        id: 'improveResponse',
        execute: improveResponseAction,
        resumeSchema: z.object({
          toneScore: z.object({ score: z.number() }),
          completenessScore: z.object({ score: z.number() }),
        }),
        inputSchema: z.object({ toneScore: z.any(), completenessScore: z.any() }),
        outputSchema: z.object({ improvedOutput: z.string() }),
      });
      const evaluateImproved = createStep({
        id: 'evaluateImprovedResponse',
        execute: evaluateImprovedAction,
        inputSchema: z.object({ improvedOutput: z.string() }),
        outputSchema: z.object({
          toneScore: z.any(),
          completenessScore: z.any(),
        }),
      });

      const promptEvalWorkflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({ input: z.string() }),
        outputSchema: z.object({}),
      });

      promptEvalWorkflow
        .then(getUserInput)
        .then(promptAgent)
        .then(evaluateTone)
        .then(improveResponse)
        .then(evaluateImproved)
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': promptEvalWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = promptEvalWorkflow.createRun();

      const initialResult = await run.start({ inputData: { input: 'test' } });
      expect(initialResult.steps.promptAgent.status).toBe('suspended');
      expect(promptAgentAction).toHaveBeenCalledTimes(1);
      // expect(initialResult.activePaths.size).toBe(1);
      // expect(initialResult.activePaths.get('promptAgent')?.status).toBe('suspended');
      // expect(initialResult.activePaths.get('promptAgent')?.suspendPayload).toEqual({ testPayload: 'hello' });
      expect(initialResult.steps).toEqual({
        input: { input: 'test' },
        getUserInput: { status: 'success', output: { userInput: 'test input' } },
        promptAgent: { status: 'suspended', payload: { testPayload: 'hello' } },
      });

      const newCtx = {
        userInput: 'test input for resumption',
      };

      expect(initialResult.steps.promptAgent.status).toBe('suspended');
      expect(promptAgentAction).toHaveBeenCalledTimes(1);

      const firstResumeResult = await run.resume({ step: 'promptAgent', resumeData: newCtx });
      if (!firstResumeResult) {
        throw new Error('Resume failed to return a result');
      }

      // expect(firstResumeResult.activePaths.size).toBe(1);
      // expect(firstResumeResult.activePaths.get('improveResponse')?.status).toBe('suspended');
      expect(firstResumeResult.steps).toEqual({
        input: { input: 'test' },
        getUserInput: { status: 'success', output: { userInput: 'test input' } },
        promptAgent: { status: 'success', output: { modelOutput: 'test output' } },
        evaluateToneConsistency: {
          status: 'success',
          output: {
            toneScore: { score: 0.8 },
            completenessScore: { score: 0.7 },
          },
        },
        improveResponse: { status: 'suspended' },
      });

      const secondResumeResult = await run.resume({
        step: improveResponse,
        resumeData: {
          toneScore: { score: 0.8 },
          completenessScore: { score: 0.7 },
        },
      });
      if (!secondResumeResult) {
        throw new Error('Resume failed to return a result');
      }

      expect(promptAgentAction).toHaveBeenCalledTimes(2);

      expect(secondResumeResult.steps).toEqual({
        input: { input: 'test' },
        getUserInput: { status: 'success', output: { userInput: 'test input' } },
        promptAgent: { status: 'success', output: { modelOutput: 'test output' } },
        evaluateToneConsistency: {
          status: 'success',
          output: { toneScore: { score: 0.8 }, completenessScore: { score: 0.7 } },
        },
        improveResponse: { status: 'success', output: { improvedOutput: 'improved output' } },
        evaluateImprovedResponse: {
          status: 'success',
          output: { toneScore: { score: 0.9 }, completenessScore: { score: 0.8 } },
        },
      });

      expect(promptAgentAction).toHaveBeenCalledTimes(2);

      srv.close();
    });
  });

  describe('Accessing Mastra', () => {
    it('should be able to access the deprecated mastra primitives', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);
      let telemetry: Telemetry | undefined;
      const step1 = createStep({
        id: 'step1',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        execute: async ({ mastra }) => {
          telemetry = mastra?.getTelemetry();
          return {};
        },
      });

      const workflow = createWorkflow({ id: 'test-workflow', inputSchema: z.object({}), outputSchema: z.object({}) });
      workflow.then(step1).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Access new instance properties directly - should work without warning
      const run = workflow.createRun();
      await run.start({ inputData: {} });

      expect(telemetry).toBeDefined();
      expect(telemetry).toBeInstanceOf(Telemetry);

      srv.close();
    });
  });

  describe('Agent as step', () => {
    it('should be able to use an agent as a step', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({
          prompt1: z.string(),
          prompt2: z.string(),
        }),
        outputSchema: z.object({}),
      });

      const agent = new Agent({
        name: 'test-agent-1',
        instructions: 'test agent instructions',
        model: openai('gpt-4'),
      });

      const agent2 = new Agent({
        name: 'test-agent-2',
        instructions: 'test agent instructions',
        model: openai('gpt-4'),
      });

      const startStep = createStep({
        id: 'start',
        inputSchema: z.object({
          prompt1: z.string(),
          prompt2: z.string(),
        }),
        outputSchema: z.object({ prompt1: z.string(), prompt2: z.string() }),
        execute: async ({ inputData }) => {
          return {
            prompt1: inputData.prompt1,
            prompt2: inputData.prompt2,
          };
        },
      });

      const agentStep1 = createStep(agent);
      const agentStep2 = createStep(agent2);

      workflow
        .then(startStep)
        .map({
          prompt: {
            step: startStep,
            path: 'prompt1',
          },
        })
        .then(agentStep1)
        .map({
          prompt: {
            step: startStep,
            path: 'prompt2',
          },
        })
        .then(agentStep2)
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({
        inputData: { prompt1: 'Capital of France, just the name', prompt2: 'Capital of UK, just the name' },
      });

      expect(result.steps['test-agent-1']).toEqual({
        status: 'success',
        output: { text: 'Paris' },
      });

      expect(result.steps['test-agent-2']).toEqual({
        status: 'success',
        output: { text: 'London' },
      });

      srv.close();
    });

    it('should be able to use an agent in parallel', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const execute = vi.fn<any>().mockResolvedValue({ result: 'success' });
      const finalStep = createStep({
        id: 'finalStep',
        inputSchema: z.object({
          'nested-workflow': z.object({ text: z.string() }),
          'nested-workflow-2': z.object({ text: z.string() }),
        }),
        outputSchema: z.object({
          result: z.string(),
        }),
        execute,
      });

      const workflow = createWorkflow({
        id: 'test-workflow',
        inputSchema: z.object({
          prompt1: z.string(),
          prompt2: z.string(),
        }),
        outputSchema: z.object({
          'nested-workflow': z.object({ text: z.string() }),
          'nested-workflow-2': z.object({ text: z.string() }),
        }),
      });

      const agent = new Agent({
        name: 'test-agent-1',
        instructions: 'test agent instructions',
        model: openai('gpt-4'),
      });

      const agent2 = new Agent({
        name: 'test-agent-2',
        instructions: 'test agent instructions',
        model: openai('gpt-4'),
      });

      const startStep = createStep({
        id: 'start',
        inputSchema: z.object({
          prompt1: z.string(),
          prompt2: z.string(),
        }),
        outputSchema: z.object({ prompt1: z.string(), prompt2: z.string() }),
        execute: async ({ inputData }) => {
          return {
            prompt1: inputData.prompt1,
            prompt2: inputData.prompt2,
          };
        },
      });

      const nestedWorkflow1 = createWorkflow({
        id: 'nested-workflow',
        inputSchema: z.object({ prompt1: z.string(), prompt2: z.string() }),
        outputSchema: z.object({ text: z.string() }),
      })
        .then(startStep)
        .map({
          prompt: {
            step: startStep,
            path: 'prompt1',
          },
        })
        .then(createStep(agent))
        .commit();

      const nestedWorkflow2 = createWorkflow({
        id: 'nested-workflow-2',
        inputSchema: z.object({ prompt1: z.string(), prompt2: z.string() }),
        outputSchema: z.object({ text: z.string() }),
      })
        .then(startStep)
        .map({
          prompt: {
            step: startStep,
            path: 'prompt2',
          },
        })
        .then(createStep(agent2))
        .commit();

      workflow.parallel([nestedWorkflow1, nestedWorkflow2]).then(finalStep).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = workflow.createRun();
      const result = await run.start({
        inputData: { prompt1: 'Capital of France, just the name', prompt2: 'Capital of UK, just the name' },
      });

      expect(execute).toHaveBeenCalledTimes(1);
      expect(result.steps['finalStep']).toEqual({
        status: 'success',
        output: { result: 'success' },
      });

      expect(result.steps['nested-workflow']).toEqual({
        status: 'success',
        output: { text: 'Paris' },
      });

      expect(result.steps['nested-workflow-2']).toEqual({
        status: 'success',
        output: { text: 'London' },
      });

      srv.close();
    });
  });

  describe('Nested workflows', () => {
    it('should be able to nest workflows', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const start = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)
        const currentValue = inputData.startValue || 0;

        // Increment the value
        const newValue = currentValue + 1;

        return { newValue };
      });
      const startStep = createStep({
        id: 'start',
        inputSchema: z.object({ startValue: z.number() }),
        outputSchema: z.object({
          newValue: z.number(),
        }),
        execute: start,
      });

      const other = vi.fn().mockImplementation(async () => {
        return { other: 26 };
      });
      const otherStep = createStep({
        id: 'other',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({ other: z.number() }),
        execute: other,
      });

      const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
        const startVal = getStepResult(startStep)?.newValue ?? 0;
        const otherVal = getStepResult(otherStep)?.other ?? 0;
        return { finalValue: startVal + otherVal };
      });
      const last = vi.fn().mockImplementation(async () => {
        return { success: true };
      });
      const finalStep = createStep({
        id: 'final',
        inputSchema: z.object({ newValue: z.number(), other: z.number() }),
        outputSchema: z.object({ success: z.boolean() }),
        execute: final,
      });

      const counterWorkflow = createWorkflow({
        id: 'counter-workflow',
        inputSchema: z.object({
          startValue: z.number(),
        }),
        outputSchema: z.object({ success: z.boolean() }),
      });

      const wfA = createWorkflow({
        id: 'nested-workflow-a',
        inputSchema: counterWorkflow.inputSchema,
        outputSchema: z.object({ success: z.boolean() }),
      })
        .then(startStep)
        .then(otherStep)
        .then(finalStep)
        .commit();
      const wfB = createWorkflow({
        id: 'nested-workflow-b',
        inputSchema: counterWorkflow.inputSchema,
        outputSchema: z.object({ success: z.boolean() }),
      })
        .then(startStep)
        .then(finalStep)
        .commit();
      counterWorkflow
        .parallel([wfA, wfB])
        .then(
          createStep({
            id: 'last-step',
            inputSchema: z.object({
              'nested-workflow-a': z.object({ success: z.boolean() }),
              'nested-workflow-b': z.object({ success: z.boolean() }),
            }),
            outputSchema: z.object({ success: z.boolean() }),
            execute: last,
          }),
        )
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { startValue: 0 } });

      srv.close();

      expect(start).toHaveBeenCalledTimes(2);
      expect(other).toHaveBeenCalledTimes(1);
      expect(final).toHaveBeenCalledTimes(2);
      expect(last).toHaveBeenCalledTimes(1);
      // @ts-ignore
      expect(result.steps['nested-workflow-a'].output).toEqual({
        finalValue: 26 + 1,
      });

      // @ts-ignore
      expect(result.steps['nested-workflow-b'].output).toEqual({
        finalValue: 1,
      });

      expect(result.steps['last-step']).toEqual({
        output: { success: true },
        status: 'success',
      });
    });

    it('should be able to nest workflows with conditions', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const start = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)
        const currentValue = inputData.startValue || 0;

        // Increment the value
        const newValue = currentValue + 1;

        return { newValue };
      });
      const startStep = createStep({
        id: 'start',
        inputSchema: z.object({ startValue: z.number() }),
        outputSchema: z.object({
          newValue: z.number(),
        }),
        execute: start,
      });

      const other = vi.fn().mockImplementation(async () => {
        return { other: 26 };
      });
      const otherStep = createStep({
        id: 'other',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({ other: z.number() }),
        execute: other,
      });

      const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
        const startVal = getStepResult(startStep)?.newValue ?? 0;
        const otherVal = getStepResult(otherStep)?.other ?? 0;
        return { finalValue: startVal + otherVal };
      });
      const last = vi.fn().mockImplementation(async () => {
        return { success: true };
      });
      const finalStep = createStep({
        id: 'final',
        inputSchema: z.object({ newValue: z.number(), other: z.number() }),
        outputSchema: z.object({ finalValue: z.number() }),
        execute: final,
      });

      const counterWorkflow = createWorkflow({
        id: 'counter-workflow',
        inputSchema: z.object({
          startValue: z.number(),
        }),
        outputSchema: z.object({ success: z.boolean() }),
      });

      const wfA = createWorkflow({
        id: 'nested-workflow-a',
        inputSchema: counterWorkflow.inputSchema,
        outputSchema: finalStep.outputSchema,
      })
        .then(startStep)
        .then(otherStep)
        .then(finalStep)
        .commit();
      const wfB = createWorkflow({
        id: 'nested-workflow-b',
        inputSchema: counterWorkflow.inputSchema,
        outputSchema: z.object({ other: otherStep.outputSchema, final: finalStep.outputSchema }),
      })
        .then(startStep)
        .branch([
          [async () => false, otherStep],
          // @ts-ignore
          [async () => true, finalStep],
        ])
        .map({
          finalValue: {
            step: finalStep,
            path: 'finalValue',
          },
        })
        .commit();
      counterWorkflow
        .parallel([wfA, wfB])
        .then(
          createStep({
            id: 'last-step',
            inputSchema: z.object({
              'nested-workflow-a': wfA.outputSchema,
              'nested-workflow-b': wfB.outputSchema,
            }),
            outputSchema: z.object({ success: z.boolean() }),
            execute: last,
          }),
        )
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { startValue: 0 } });

      srv.close();

      expect(start).toHaveBeenCalledTimes(2);
      expect(other).toHaveBeenCalledTimes(1);
      expect(final).toHaveBeenCalledTimes(2);
      expect(last).toHaveBeenCalledTimes(1);
      // @ts-ignore
      expect(result.steps['nested-workflow-a'].output).toEqual({
        finalValue: 26 + 1,
      });

      // @ts-ignore
      expect(result.steps['nested-workflow-b'].output).toEqual({
        finalValue: 1,
      });

      expect(result.steps['last-step']).toEqual({
        output: { success: true },
        status: 'success',
      });
    });

    describe('new if else branching syntax with nested workflows', () => {
      it('should execute if-branch', async ctx => {
        const inngest = new Inngest({
          id: 'mastra',
          baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        });

        const { createWorkflow, createStep } = init(inngest);

        const start = vi.fn().mockImplementation(async ({ inputData }) => {
          // Get the current value (either from trigger or previous increment)
          const currentValue = inputData.startValue || 0;

          // Increment the value
          const newValue = currentValue + 1;

          return { newValue };
        });
        const startStep = createStep({
          id: 'start',
          inputSchema: z.object({ startValue: z.number() }),
          outputSchema: z.object({
            newValue: z.number(),
          }),
          execute: start,
        });

        const other = vi.fn().mockImplementation(async () => {
          return { other: 26 };
        });
        const otherStep = createStep({
          id: 'other',
          inputSchema: z.object({ newValue: z.number() }),
          outputSchema: z.object({ other: z.number() }),
          execute: other,
        });

        const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
          const startVal = getStepResult(startStep)?.newValue ?? 0;
          const otherVal = getStepResult(otherStep)?.other ?? 0;
          return { finalValue: startVal + otherVal };
        });
        const first = vi.fn().mockImplementation(async () => {
          return { success: true };
        });
        const last = vi.fn().mockImplementation(async () => {
          return { success: true };
        });
        const finalStep = createStep({
          id: 'final',
          inputSchema: z.object({ newValue: z.number(), other: z.number() }),
          outputSchema: z.object({ finalValue: z.number() }),
          execute: final,
        });

        const counterWorkflow = createWorkflow({
          id: 'counter-workflow',
          inputSchema: z.object({
            startValue: z.number(),
          }),
          outputSchema: z.object({ success: z.boolean() }),
        });

        const wfA = createWorkflow({
          id: 'nested-workflow-a',
          inputSchema: counterWorkflow.inputSchema,
          outputSchema: finalStep.outputSchema,
        })
          .then(startStep)
          .then(otherStep)
          .then(finalStep)
          .commit();
        const wfB = createWorkflow({
          id: 'nested-workflow-b',
          inputSchema: counterWorkflow.inputSchema,
          outputSchema: finalStep.outputSchema,
        })
          .then(startStep)
          .then(finalStep)
          .commit();
        counterWorkflow
          .then(
            createStep({
              id: 'first-step',
              inputSchema: z.object({ startValue: z.number() }),
              outputSchema: wfA.inputSchema,
              execute: first,
            }),
          )
          .branch([
            [async () => true, wfA],
            [async () => false, wfB],
          ])
          .then(
            createStep({
              id: 'last-step',
              inputSchema: z.object({
                'nested-workflow-a': wfA.outputSchema,
                'nested-workflow-b': wfB.outputSchema,
              }),
              outputSchema: z.object({ success: z.boolean() }),
              execute: last,
            }),
          )
          .commit();

        const mastra = new Mastra({
          storage: new DefaultStorage({
            url: ':memory:',
          }),
          workflows: {
            'test-workflow': counterWorkflow,
          },
          server: {
            apiRoutes: [
              {
                path: '/inngest/api',
                method: 'ALL',
                createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
              },
            ],
          },
        });

        const app = await createHonoServer(mastra);
        app.use('*', async (ctx, next) => {
          await next();
        });

        const srv = serve({
          fetch: app.fetch,
          port: (ctx as any).handlerPort,
        });

        const run = counterWorkflow.createRun();
        const result = await run.start({ inputData: { startValue: 0 } });

        srv.close();

        expect(start).toHaveBeenCalledTimes(1);
        expect(other).toHaveBeenCalledTimes(1);
        expect(final).toHaveBeenCalledTimes(1);
        expect(first).toHaveBeenCalledTimes(1);
        expect(last).toHaveBeenCalledTimes(1);
        // @ts-ignore
        expect(result.steps['nested-workflow-a'].output).toEqual({
          finalValue: 26 + 1,
        });

        expect(result.steps['first-step']).toEqual({
          output: { success: true },
          status: 'success',
        });

        expect(result.steps['last-step']).toEqual({
          output: { success: true },
          status: 'success',
        });
      });

      it('should execute else-branch', async ctx => {
        const inngest = new Inngest({
          id: 'mastra',
          baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        });

        const { createWorkflow, createStep } = init(inngest);

        const start = vi.fn().mockImplementation(async ({ inputData }) => {
          // Get the current value (either from trigger or previous increment)
          const currentValue = inputData.startValue || 0;

          // Increment the value
          const newValue = currentValue + 1;

          return { newValue };
        });
        const startStep = createStep({
          id: 'start',
          inputSchema: z.object({ startValue: z.number() }),
          outputSchema: z.object({
            newValue: z.number(),
          }),
          execute: start,
        });

        const other = vi.fn().mockImplementation(async () => {
          return { other: 26 };
        });
        const otherStep = createStep({
          id: 'other',
          inputSchema: z.object({ newValue: z.number() }),
          outputSchema: z.object({ other: z.number() }),
          execute: other,
        });

        const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
          const startVal = getStepResult(startStep)?.newValue ?? 0;
          const otherVal = getStepResult(otherStep)?.other ?? 0;
          return { finalValue: startVal + otherVal };
        });
        const first = vi.fn().mockImplementation(async () => {
          return { success: true };
        });
        const last = vi.fn().mockImplementation(async () => {
          return { success: true };
        });
        const finalStep = createStep({
          id: 'final',
          inputSchema: z.object({ newValue: z.number(), other: z.number() }),
          outputSchema: z.object({ finalValue: z.number() }),
          execute: final,
        });

        const counterWorkflow = createWorkflow({
          id: 'counter-workflow',
          inputSchema: z.object({
            startValue: z.number(),
          }),
          outputSchema: z.object({ success: z.boolean() }),
        });

        const wfA = createWorkflow({
          id: 'nested-workflow-a',
          inputSchema: counterWorkflow.inputSchema,
          outputSchema: finalStep.outputSchema,
        })
          .then(startStep)
          .then(otherStep)
          .then(finalStep)
          .commit();
        const wfB = createWorkflow({
          id: 'nested-workflow-b',
          inputSchema: counterWorkflow.inputSchema,
          outputSchema: finalStep.outputSchema,
        })
          .then(startStep)
          .then(finalStep)
          .commit();
        counterWorkflow
          .then(
            createStep({
              id: 'first-step',
              inputSchema: z.object({ startValue: z.number() }),
              outputSchema: wfA.inputSchema,
              execute: first,
            }),
          )
          .branch([
            [async () => false, wfA],
            [async () => true, wfB],
          ])
          .then(
            createStep({
              id: 'last-step',
              inputSchema: z.object({
                'nested-workflow-a': wfA.outputSchema,
                'nested-workflow-b': wfB.outputSchema,
              }),
              outputSchema: z.object({ success: z.boolean() }),
              execute: last,
            }),
          )
          .commit();

        const mastra = new Mastra({
          storage: new DefaultStorage({
            url: ':memory:',
          }),
          workflows: {
            'test-workflow': counterWorkflow,
          },
          server: {
            apiRoutes: [
              {
                path: '/inngest/api',
                method: 'ALL',
                createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
              },
            ],
          },
        });

        const app = await createHonoServer(mastra);
        app.use('*', async (ctx, next) => {
          await next();
        });

        const srv = serve({
          fetch: app.fetch,
          port: (ctx as any).handlerPort,
        });

        const run = counterWorkflow.createRun();
        const result = await run.start({ inputData: { startValue: 0 } });

        srv.close();

        expect(start).toHaveBeenCalledTimes(1);
        expect(other).toHaveBeenCalledTimes(0);
        expect(final).toHaveBeenCalledTimes(1);
        expect(first).toHaveBeenCalledTimes(1);
        expect(last).toHaveBeenCalledTimes(1);

        // @ts-ignore
        expect(result.steps['nested-workflow-b'].output).toEqual({
          finalValue: 1,
        });

        expect(result.steps['first-step']).toEqual({
          output: { success: true },
          status: 'success',
        });

        expect(result.steps['last-step']).toEqual({
          output: { success: true },
          status: 'success',
        });
      });

      it('should execute nested else and if-branch', async ctx => {
        const inngest = new Inngest({
          id: 'mastra',
          baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        });

        const { createWorkflow, createStep } = init(inngest);

        const start = vi.fn().mockImplementation(async ({ inputData }) => {
          // Get the current value (either from trigger or previous increment)
          const currentValue = inputData.startValue || 0;

          // Increment the value
          const newValue = currentValue + 1;

          return { newValue };
        });
        const startStep = createStep({
          id: 'start',
          inputSchema: z.object({ startValue: z.number() }),
          outputSchema: z.object({
            newValue: z.number(),
          }),
          execute: start,
        });

        const other = vi.fn().mockImplementation(async () => {
          return { other: 26 };
        });
        const otherStep = createStep({
          id: 'other',
          inputSchema: z.object({ newValue: z.number() }),
          outputSchema: z.object({ other: z.number() }),
          execute: other,
        });

        const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
          const startVal = getStepResult(startStep)?.newValue ?? 0;
          const otherVal = getStepResult(otherStep)?.other ?? 0;
          return { finalValue: startVal + otherVal };
        });
        const first = vi.fn().mockImplementation(async () => {
          return { success: true };
        });
        const last = vi.fn().mockImplementation(async () => {
          return { success: true };
        });
        const finalStep = createStep({
          id: 'final',
          inputSchema: z.object({ newValue: z.number(), other: z.number() }),
          outputSchema: z.object({ finalValue: z.number() }),
          execute: final,
        });

        const counterWorkflow = createWorkflow({
          id: 'counter-workflow',
          inputSchema: z.object({
            startValue: z.number(),
          }),
          outputSchema: z.object({ success: z.boolean() }),
        });

        const wfA = createWorkflow({
          id: 'nested-workflow-a',
          inputSchema: counterWorkflow.inputSchema,
          outputSchema: finalStep.outputSchema,
        })
          .then(startStep)
          .then(otherStep)
          .then(finalStep)
          .commit();
        const wfB = createWorkflow({
          id: 'nested-workflow-b',
          inputSchema: counterWorkflow.inputSchema,
          outputSchema: finalStep.outputSchema,
        })
          .then(startStep)
          .branch([
            [
              async () => true,
              createWorkflow({
                id: 'nested-workflow-c',
                inputSchema: startStep.outputSchema,
                outputSchema: otherStep.outputSchema,
              })
                .then(otherStep)
                .commit(),
            ],
            [
              async () => false,
              createWorkflow({
                id: 'nested-workflow-d',
                inputSchema: startStep.outputSchema,
                outputSchema: otherStep.outputSchema,
              })
                .then(otherStep)
                .commit(),
            ],
          ])
          // TODO: maybe make this a little nicer to do with .map()?
          .then(
            createStep({
              id: 'map-results',
              inputSchema: z.object({
                'nested-workflow-c': otherStep.outputSchema,
                'nested-workflow-d': otherStep.outputSchema,
              }),
              outputSchema: otherStep.outputSchema,
              execute: async ({ inputData }) => {
                return { other: inputData['nested-workflow-c']?.other ?? inputData['nested-workflow-d']?.other };
              },
            }),
          )
          .then(finalStep)
          .commit();

        counterWorkflow
          .then(
            createStep({
              id: 'first-step',
              inputSchema: z.object({ startValue: z.number() }),
              outputSchema: wfA.inputSchema,
              execute: first,
            }),
          )
          .branch([
            [async () => false, wfA],
            [async () => true, wfB],
          ])
          .then(
            createStep({
              id: 'last-step',
              inputSchema: z.object({
                'nested-workflow-a': wfA.outputSchema,
                'nested-workflow-b': wfB.outputSchema,
              }),
              outputSchema: z.object({ success: z.boolean() }),
              execute: last,
            }),
          )
          .commit();

        const mastra = new Mastra({
          storage: new DefaultStorage({
            url: ':memory:',
          }),
          workflows: {
            'test-workflow': counterWorkflow,
          },
          server: {
            apiRoutes: [
              {
                path: '/inngest/api',
                method: 'ALL',
                createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
              },
            ],
          },
        });

        const app = await createHonoServer(mastra);
        app.use('*', async (ctx, next) => {
          await next();
        });

        const srv = serve({
          fetch: app.fetch,
          port: (ctx as any).handlerPort,
        });

        const run = counterWorkflow.createRun();
        const result = await run.start({ inputData: { startValue: 1 } });

        srv.close();

        // expect(start).toHaveBeenCalledTimes(1);
        // expect(other).toHaveBeenCalledTimes(1);
        // expect(final).toHaveBeenCalledTimes(1);
        // expect(first).toHaveBeenCalledTimes(1);
        // expect(last).toHaveBeenCalledTimes(1);

        // @ts-ignore
        expect(result.steps['nested-workflow-b'].output).toEqual({
          finalValue: 1,
        });

        expect(result.steps['first-step']).toEqual({
          output: { success: true },
          status: 'success',
        });

        expect(result.steps['last-step']).toEqual({
          output: { success: true },
          status: 'success',
        });
      });
    });

    describe('suspending and resuming nested workflows', () => {
      it('should be able to suspend nested workflow step', async ctx => {
        const inngest = new Inngest({
          id: 'mastra',
          baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        });

        const { createWorkflow, createStep } = init(inngest);

        const start = vi.fn().mockImplementation(async ({ inputData }) => {
          // Get the current value (either from trigger or previous increment)
          const currentValue = inputData.startValue || 0;

          // Increment the value
          const newValue = currentValue + 1;

          return { newValue };
        });
        const startStep = createStep({
          id: 'start',
          inputSchema: z.object({ startValue: z.number() }),
          outputSchema: z.object({
            newValue: z.number(),
          }),
          execute: start,
        });

        const other = vi.fn().mockImplementation(async ({ suspend, resumeData }) => {
          if (!resumeData) {
            await suspend();
          }
          return { other: 26 };
        });
        const otherStep = createStep({
          id: 'other',
          inputSchema: z.object({ newValue: z.number() }),
          outputSchema: z.object({ other: z.number() }),
          execute: other,
        });

        const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
          const startVal = getStepResult(startStep)?.newValue ?? 0;
          const otherVal = getStepResult(otherStep)?.other ?? 0;
          return { finalValue: startVal + otherVal };
        });
        const last = vi.fn().mockImplementation(async ({}) => {
          return { success: true };
        });
        const begin = vi.fn().mockImplementation(async ({ inputData }) => {
          return inputData;
        });
        const finalStep = createStep({
          id: 'final',
          inputSchema: z.object({ newValue: z.number(), other: z.number() }),
          outputSchema: z.object({
            finalValue: z.number(),
          }),
          execute: final,
        });

        const counterWorkflow = createWorkflow({
          id: 'counter-workflow',
          inputSchema: z.object({
            startValue: z.number(),
          }),
          outputSchema: z.object({
            finalValue: z.number(),
          }),
        });

        const wfA = createWorkflow({
          id: 'nested-workflow-a',
          inputSchema: counterWorkflow.inputSchema,
          outputSchema: finalStep.outputSchema,
        })
          .then(startStep)
          .then(otherStep)
          .then(finalStep)
          .commit();

        counterWorkflow
          .then(
            createStep({
              id: 'begin-step',
              inputSchema: counterWorkflow.inputSchema,
              outputSchema: counterWorkflow.inputSchema,
              execute: begin,
            }),
          )
          .then(wfA)
          .then(
            createStep({
              id: 'last-step',
              inputSchema: wfA.outputSchema,
              outputSchema: z.object({ success: z.boolean() }),
              execute: last,
            }),
          )
          .commit();

        const mastra = new Mastra({
          storage: new DefaultStorage({
            url: ':memory:',
          }),
          workflows: {
            'test-workflow': counterWorkflow,
          },
          server: {
            apiRoutes: [
              {
                path: '/inngest/api',
                method: 'ALL',
                createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
              },
            ],
          },
        });

        const app = await createHonoServer(mastra);

        const srv = serve({
          fetch: app.fetch,
          port: (ctx as any).handlerPort,
        });
        await new Promise(resolve => setTimeout(resolve, 2000));

        const run = counterWorkflow.createRun();
        const result = await run.start({ inputData: { startValue: 0 } });

        expect(begin).toHaveBeenCalledTimes(1);
        expect(start).toHaveBeenCalledTimes(1);
        expect(other).toHaveBeenCalledTimes(1);
        expect(final).toHaveBeenCalledTimes(0);
        expect(last).toHaveBeenCalledTimes(0);
        expect(result.steps['nested-workflow-a']).toMatchObject({
          status: 'suspended',
        });

        // @ts-ignore
        expect(result.steps['last-step']).toEqual(undefined);

        const resumedResults = await run.resume({ step: [wfA, otherStep], resumeData: { newValue: 0 } });

        // @ts-ignore
        expect(resumedResults.steps['nested-workflow-a'].output).toEqual({
          finalValue: 26 + 1,
        });

        expect(start).toHaveBeenCalledTimes(1);
        expect(other).toHaveBeenCalledTimes(2);
        expect(final).toHaveBeenCalledTimes(1);
        expect(last).toHaveBeenCalledTimes(1);

        srv.close();
      });
    });

    describe('Workflow results', () => {
      it('should be able to spec out workflow result via variables', async ctx => {
        const inngest = new Inngest({
          id: 'mastra',
          baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
        });

        const { createWorkflow, createStep } = init(inngest);

        const start = vi.fn().mockImplementation(async ({ inputData }) => {
          // Get the current value (either from trigger or previous increment)
          const currentValue = inputData.startValue || 0;

          // Increment the value
          const newValue = currentValue + 1;

          return { newValue };
        });
        const startStep = createStep({
          id: 'start',
          inputSchema: z.object({ startValue: z.number() }),
          outputSchema: z.object({
            newValue: z.number(),
          }),
          execute: start,
        });

        const other = vi.fn().mockImplementation(async () => {
          return { other: 26 };
        });
        const otherStep = createStep({
          id: 'other',
          inputSchema: z.object({ newValue: z.number() }),
          outputSchema: z.object({ other: z.number() }),
          execute: other,
        });

        const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
          const startVal = getStepResult(startStep)?.newValue ?? 0;
          const otherVal = getStepResult(otherStep)?.other ?? 0;
          return { finalValue: startVal + otherVal };
        });
        const last = vi.fn().mockImplementation(async () => {
          return { success: true };
        });
        const finalStep = createStep({
          id: 'final',
          inputSchema: z.object({ newValue: z.number(), other: z.number() }),
          outputSchema: z.object({
            finalValue: z.number(),
          }),
          execute: final,
        });

        const wfA = createWorkflow({
          steps: [startStep, otherStep, finalStep],
          id: 'nested-workflow-a',
          inputSchema: z.object({
            startValue: z.number(),
          }),
          outputSchema: z.object({
            finalValue: z.number(),
          }),
        })
          .then(startStep)
          .then(otherStep)
          .then(finalStep)
          .commit();

        const counterWorkflow = createWorkflow({
          id: 'counter-workflow',
          inputSchema: z.object({
            startValue: z.number(),
          }),
          outputSchema: z.object({
            finalValue: z.number(),
          }),
        });

        counterWorkflow
          .then(wfA)
          .then(
            createStep({
              id: 'last-step',
              inputSchema: wfA.outputSchema,
              outputSchema: z.object({ success: z.boolean() }),
              execute: last,
            }),
          )
          .commit();

        const mastra = new Mastra({
          storage: new DefaultStorage({
            url: ':memory:',
          }),
          workflows: {
            'test-workflow': counterWorkflow,
          },
          server: {
            apiRoutes: [
              {
                path: '/inngest/api',
                method: 'ALL',
                createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
              },
            ],
          },
        });

        const app = await createHonoServer(mastra);
        app.use('*', async (ctx, next) => {
          await next();
        });

        const srv = serve({
          fetch: app.fetch,
          port: (ctx as any).handlerPort,
        });

        const run = counterWorkflow.createRun();
        const result = await run.start({ inputData: { startValue: 0 } });
        const results = result.steps;

        srv.close();

        expect(start).toHaveBeenCalledTimes(1);
        expect(other).toHaveBeenCalledTimes(1);
        expect(final).toHaveBeenCalledTimes(1);
        expect(last).toHaveBeenCalledTimes(1);

        // @ts-ignore
        expect(results['nested-workflow-a']).toMatchObject({
          status: 'success',
          output: {
            finalValue: 26 + 1,
          },
        });

        expect(result.steps['last-step']).toEqual({
          status: 'success',
          output: { success: true },
        });
      });
    });

    it('should be able to suspend nested workflow step in a nested workflow step', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const start = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)
        const currentValue = inputData.startValue || 0;

        // Increment the value
        const newValue = currentValue + 1;

        return { newValue };
      });
      const startStep = createStep({
        id: 'start',
        inputSchema: z.object({ startValue: z.number() }),
        outputSchema: z.object({
          newValue: z.number(),
        }),
        execute: start,
      });

      const other = vi.fn().mockImplementation(async ({ suspend, resumeData }) => {
        if (!resumeData) {
          await suspend();
        }
        return { other: 26 };
      });
      const otherStep = createStep({
        id: 'other',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({ other: z.number() }),
        execute: other,
      });

      const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
        const startVal = getStepResult(startStep)?.newValue ?? 0;
        const otherVal = getStepResult(otherStep)?.other ?? 0;
        return { finalValue: startVal + otherVal };
      });
      const last = vi.fn().mockImplementation(async ({}) => {
        return { success: true };
      });
      const begin = vi.fn().mockImplementation(async ({ inputData }) => {
        return inputData;
      });
      const finalStep = createStep({
        id: 'final',
        inputSchema: z.object({ newValue: z.number(), other: z.number() }),
        outputSchema: z.object({
          finalValue: z.number(),
        }),
        execute: final,
      });

      const counterInputSchema = z.object({
        startValue: z.number(),
      });
      const counterOutputSchema = z.object({
        finalValue: z.number(),
      });

      const passthroughStep = createStep({
        id: 'passthrough',
        inputSchema: counterInputSchema,
        outputSchema: counterInputSchema,
        execute: vi.fn().mockImplementation(async ({ inputData }) => {
          return inputData;
        }),
      });

      const wfA = createWorkflow({
        id: 'nested-workflow-a',
        inputSchema: counterInputSchema,
        outputSchema: finalStep.outputSchema,
      })
        .then(startStep)
        .then(otherStep)
        .then(finalStep)
        .commit();

      const wfB = createWorkflow({
        id: 'nested-workflow-b',
        inputSchema: counterInputSchema,
        outputSchema: finalStep.outputSchema,
      })
        .then(passthroughStep)
        .then(wfA)
        .commit();

      const wfC = createWorkflow({
        id: 'nested-workflow-c',
        inputSchema: counterInputSchema,
        outputSchema: finalStep.outputSchema,
      })
        .then(passthroughStep)
        .then(wfB)
        .commit();

      const counterWorkflow = createWorkflow({
        id: 'counter-workflow',
        inputSchema: counterInputSchema,
        outputSchema: counterOutputSchema,
        steps: [wfC, passthroughStep],
      });

      counterWorkflow
        .then(
          createStep({
            id: 'begin-step',
            inputSchema: counterWorkflow.inputSchema,
            outputSchema: counterWorkflow.inputSchema,
            execute: begin,
          }),
        )
        .then(wfC)
        .then(
          createStep({
            id: 'last-step',
            inputSchema: wfA.outputSchema,
            outputSchema: z.object({ success: z.boolean() }),
            execute: last,
          }),
        )
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { startValue: 0 } });

      expect(passthroughStep.execute).toHaveBeenCalledTimes(2);
      expect(result.steps['nested-workflow-c']).toMatchObject({
        status: 'suspended',
        payload: {
          __workflow_meta: {
            path: ['nested-workflow-b', 'nested-workflow-a', 'other'],
          },
        },
      });

      // @ts-ignore
      expect(result.steps['last-step']).toEqual(undefined);

      if (result.status !== 'suspended') {
        expect.fail('Workflow should be suspended');
      }
      expect(result.suspended[0]).toEqual(['nested-workflow-c', 'nested-workflow-b', 'nested-workflow-a', 'other']);
      const resumedResults = await run.resume({ step: result.suspended[0], resumeData: { newValue: 0 } });

      srv.close();

      // @ts-ignore
      expect(resumedResults.steps['nested-workflow-c'].output).toEqual({
        finalValue: 26 + 1,
      });

      expect(start).toHaveBeenCalledTimes(1);
      expect(other).toHaveBeenCalledTimes(2);
      expect(final).toHaveBeenCalledTimes(1);
      expect(last).toHaveBeenCalledTimes(1);
      expect(passthroughStep.execute).toHaveBeenCalledTimes(2);
    });

    it('should be able clone workflows as steps', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep, cloneStep, cloneWorkflow } = init(inngest);

      const start = vi.fn().mockImplementation(async ({ inputData }) => {
        // Get the current value (either from trigger or previous increment)
        const currentValue = inputData.startValue || 0;

        // Increment the value
        const newValue = currentValue + 1;

        return { newValue };
      });
      const startStep = createStep({
        id: 'start',
        inputSchema: z.object({ startValue: z.number() }),
        outputSchema: z.object({
          newValue: z.number(),
        }),
        execute: start,
      });

      const other = vi.fn().mockImplementation(async () => {
        return { other: 26 };
      });
      const otherStep = createStep({
        id: 'other',
        inputSchema: z.object({ newValue: z.number() }),
        outputSchema: z.object({ other: z.number() }),
        execute: other,
      });

      const final = vi.fn().mockImplementation(async ({ getStepResult }) => {
        const startVal = getStepResult(startStep)?.newValue ?? 0;
        const otherVal = getStepResult(cloneStep(otherStep, { id: 'other-clone' }))?.other ?? 0;
        return { finalValue: startVal + otherVal };
      });
      const last = vi.fn().mockImplementation(async ({ inputData }) => {
        console.log('inputData', inputData);
        return { success: true };
      });
      const finalStep = createStep({
        id: 'final',
        inputSchema: z.object({ newValue: z.number(), other: z.number() }),
        outputSchema: z.object({ success: z.boolean() }),
        execute: final,
      });

      const counterWorkflow = createWorkflow({
        id: 'counter-workflow',
        inputSchema: z.object({
          startValue: z.number(),
        }),
        outputSchema: z.object({ success: z.boolean() }),
      });

      const wfA = createWorkflow({
        id: 'nested-workflow-a',
        inputSchema: counterWorkflow.inputSchema,
        outputSchema: z.object({ success: z.boolean() }),
      })
        .then(startStep)
        .then(cloneStep(otherStep, { id: 'other-clone' }))
        .then(finalStep)
        .commit();
      const wfB = createWorkflow({
        id: 'nested-workflow-b',
        inputSchema: counterWorkflow.inputSchema,
        outputSchema: z.object({ success: z.boolean() }),
      })
        .then(startStep)
        .then(cloneStep(finalStep, { id: 'final-clone' }))
        .commit();

      const wfAClone = cloneWorkflow(wfA, { id: 'nested-workflow-a-clone' });

      counterWorkflow
        .parallel([wfAClone, wfB])
        .then(
          createStep({
            id: 'last-step',
            inputSchema: z.object({
              'nested-workflow-b': z.object({ success: z.boolean() }),
              'nested-workflow-a-clone': z.object({ success: z.boolean() }),
            }),
            outputSchema: z.object({ success: z.boolean() }),
            execute: last,
          }),
        )
        .commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': counterWorkflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      const run = counterWorkflow.createRun();
      const result = await run.start({ inputData: { startValue: 0 } });

      srv.close();

      expect(start).toHaveBeenCalledTimes(2);
      expect(other).toHaveBeenCalledTimes(1);
      expect(final).toHaveBeenCalledTimes(2);
      expect(last).toHaveBeenCalledTimes(1);
      // @ts-ignore
      expect(result.steps['nested-workflow-a-clone'].output).toEqual({
        finalValue: 26 + 1,
      });

      // @ts-ignore
      expect(result.steps['nested-workflow-b'].output).toEqual({
        finalValue: 1,
      });

      expect(result.steps['last-step']).toEqual({
        output: { success: true },
        status: 'success',
      });
    });
  });

  describe('Accessing Mastra', () => {
    it('should be able to access the deprecated mastra primitives', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      let telemetry: Telemetry | undefined;
      const step1 = createStep({
        id: 'step1',
        inputSchema: z.object({}),
        outputSchema: z.object({}),
        execute: async ({ mastra }) => {
          telemetry = mastra?.getTelemetry();
          return {};
        },
      });

      const workflow = createWorkflow({ id: 'test-workflow', inputSchema: z.object({}), outputSchema: z.object({}) });
      workflow.then(step1).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Access new instance properties directly - should work without warning
      const run = workflow.createRun();
      await run.start({ inputData: {} });

      srv.close();

      expect(telemetry).toBeDefined();
      expect(telemetry).toBeInstanceOf(Telemetry);
    });
  });

  // TODO: can we support this on inngest?
  describe.skip('Dependency Injection', () => {
    it('should inject runtimeContext dependencies into steps during run', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const runtimeContext = new RuntimeContext();
      const testValue = 'test-dependency';
      runtimeContext.set('testKey', testValue);

      const step = createStep({
        id: 'step1',
        execute: async ({ runtimeContext }) => {
          const value = runtimeContext.get('testKey');
          return { injectedValue: value };
        },
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      const workflow = createWorkflow({ id: 'test-workflow', inputSchema: z.object({}), outputSchema: z.object({}) });
      workflow.then(step).commit();

      const mastra = new Mastra({
        storage: new DefaultStorage({
          url: ':memory:',
        }),
        workflows: {
          'test-workflow': workflow,
        },
        server: {
          apiRoutes: [
            {
              path: '/inngest/api',
              method: 'ALL',
              createHandler: async ({ mastra }) => inngestServe({ mastra, inngest }),
            },
          ],
        },
      });

      const app = await createHonoServer(mastra);

      const srv = serve({
        fetch: app.fetch,
        port: (ctx as any).handlerPort,
      });

      const run = workflow.createRun();
      const result = await run.start({ runtimeContext });

      srv.close();

      // @ts-ignore
      expect(result.steps.step1.output.injectedValue).toBe(testValue);
    });

    it('should inject runtimeContext dependencies into steps during resume', async ctx => {
      const inngest = new Inngest({
        id: 'mastra',
        baseUrl: `http://localhost:${(ctx as any).inngestPort}`,
      });

      const { createWorkflow, createStep } = init(inngest);

      const initialStorage = new DefaultStorage({
        url: 'file::memory:',
      });

      const runtimeContext = new RuntimeContext();
      const testValue = 'test-dependency';
      runtimeContext.set('testKey', testValue);

      const mastra = new Mastra({
        logger: false,
        storage: initialStorage,
      });

      const execute = vi.fn(async ({ runtimeContext, suspend, resumeData }) => {
        if (!resumeData?.human) {
          await suspend();
        }

        const value = runtimeContext.get('testKey');
        return { injectedValue: value };
      });

      const step = createStep({
        id: 'step1',
        execute,
        inputSchema: z.object({ human: z.boolean() }),
        outputSchema: z.object({}),
      });
      const workflow = createWorkflow({
        id: 'test-workflow',
        mastra,
        inputSchema: z.object({}),
        outputSchema: z.object({}),
      });
      workflow.then(step).commit();

      const run = workflow.createRun();
      await run.start({ runtimeContext });

      const resumeruntimeContext = new RuntimeContext();
      resumeruntimeContext.set('testKey', testValue + '2');

      const result = await run.resume({
        step: step,
        resumeData: {
          human: true,
        },
        runtimeContext: resumeruntimeContext,
      });

      // @ts-ignore
      expect(result?.steps.step1.output.injectedValue).toBe(testValue + '2');
    });
  });

  describe('Access to inngest step primitives', () => {});
}, 40e3);
