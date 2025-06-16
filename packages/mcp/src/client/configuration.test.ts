import { spawn } from 'child_process';
import path from 'path';
import { openai } from '@ai-sdk/openai';
import { Agent } from '@mastra/core/agent';
import { RuntimeContext } from '@mastra/core/di';
import type { ResourceTemplate } from '@modelcontextprotocol/sdk/types.js';
import { describe, it, expect, beforeEach, afterEach, afterAll, beforeAll, vi } from 'vitest';
import { allTools, mcpServerName } from '../__fixtures__/fire-crawl-complex-schema';
import type { LogHandler, LogMessage } from './client';
import { MCPClient } from './configuration';

vi.setConfig({ testTimeout: 80000, hookTimeout: 80000 });

describe('MCPClient', () => {
  let mcp: MCPClient;
  let weatherProcess: ReturnType<typeof spawn>;
  let clients: MCPClient[] = [];
  let weatherServerPort: number;

  beforeAll(async () => {
    weatherServerPort = 60000 + Math.floor(Math.random() * 1000); // Generate a random port
    // Start the weather SSE server
    weatherProcess = spawn('npx', ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/weather.ts')], {
      env: { ...process.env, WEATHER_SERVER_PORT: String(weatherServerPort) }, // Pass port as env var
    });

    // Wait for SSE server to be ready
    let resolved = false;
    await new Promise<void>((resolve, reject) => {
      weatherProcess.on(`exit`, () => {
        if (!resolved) reject();
      });
      if (weatherProcess.stderr) {
        weatherProcess.stderr.on(`data`, chunk => {
          console.error(chunk.toString());
        });
      }
      if (weatherProcess.stdout) {
        weatherProcess.stdout.on('data', chunk => {
          if (chunk.toString().includes('server is running on SSE')) {
            resolve();
            resolved = true;
          }
        });
      }
    });
  });

  beforeEach(async () => {
    // Give each MCPClient a unique ID to prevent re-initialization errors across tests
    const testId = "testId"
    mcp = new MCPClient({
      id: testId,
      servers: {
        stockPrice: {
          command: 'npx',
          args: ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
          env: {
            FAKE_CREDS: 'test',
          },
        },
        weather: {
          url: new URL(`http://localhost:${weatherServerPort}/sse`), // Use the dynamic port
        },
      },
    });
    clients.push(mcp);
  });

  afterEach(async () => {
    // Clean up any connected clients
    await mcp.disconnect();
    const index = clients.indexOf(mcp);
    if (index > -1) {
      clients.splice(index, 1);
    }
  });

  afterAll(async () => {
    // Kill the weather SSE server
    weatherProcess.kill('SIGINT');
  });

  describe('Instance Management', () => {
    it('should initialize with server configurations', () => {
      expect(mcp['serverConfigs']).toEqual({
        stockPrice: {
          command: 'npx',
          args: ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
        env: {
          FAKE_CREDS: 'test',
        },
        },
        weather: {
          url: new URL(`http://localhost:${weatherServerPort}/sse`),
        },
      });
    });

    it('should get connected tools with namespaced tool names', async () => {
      const connectedTools = await mcp.getTools();

      // Each tool should be namespaced with its server name
      expect(connectedTools).toHaveProperty('stockPrice_getStockPrice');
      expect(connectedTools).toHaveProperty('weather_getWeather');
    });

    it('should get connected toolsets grouped by server', async () => {
      const connectedToolsets = await mcp.getToolsets();

      expect(connectedToolsets).toHaveProperty('stockPrice');
      expect(connectedToolsets).toHaveProperty('weather');
      expect(connectedToolsets.stockPrice).toHaveProperty('getStockPrice');
      expect(connectedToolsets.weather).toHaveProperty('getWeather');
    });
  });

  describe('Resources', () => {
    it('should get resources from connected MCP servers', async () => {
      const resources = await mcp.resources.list();

      expect(resources).toHaveProperty('weather');
      expect(resources.weather).toBeDefined();
      expect(resources.weather).toHaveLength(3);

      // Verify that each expected resource exists with the correct structure
      const weatherResources = resources.weather;
      const currentWeather = weatherResources.find(r => r.uri === 'weather://current');
      expect(currentWeather).toBeDefined();
      expect(currentWeather).toMatchObject({
        uri: 'weather://current',
        name: 'Current Weather Data',
        description: expect.any(String),
        mimeType: 'application/json',
      });

      const forecast = weatherResources.find(r => r.uri === 'weather://forecast');
      expect(forecast).toBeDefined();
      expect(forecast).toMatchObject({
        uri: 'weather://forecast',
        name: 'Weather Forecast',
        description: expect.any(String),
        mimeType: 'application/json',
      });

      const historical = weatherResources.find(r => r.uri === 'weather://historical');
      expect(historical).toBeDefined();
      expect(historical).toMatchObject({
        uri: 'weather://historical',
        name: 'Historical Weather Data',
        description: expect.any(String),
        mimeType: 'application/json',
      });
    });

    it('should list resource templates from connected MCP servers', async () => {
      const templates = await mcp.resources.templates();
      expect(templates).toHaveProperty('weather');
      expect(templates.weather).toBeDefined();
      expect(templates.weather.length).toBeGreaterThan(0);
      const customForecastTemplate = templates.weather.find(
        (t: ResourceTemplate) => t.uriTemplate === 'weather://custom/{city}/{days}',
      );
      expect(customForecastTemplate).toBeDefined();
      expect(customForecastTemplate).toMatchObject({
        uriTemplate: 'weather://custom/{city}/{days}',
        name: 'Custom Weather Forecast',
        description: expect.any(String),
        mimeType: 'application/json',
      });
    });

    it('should read a specific resource from a server', async () => {
      const resourceContent = await mcp.resources.read('weather', 'weather://current');
      expect(resourceContent).toBeDefined();
      expect(resourceContent.contents).toBeInstanceOf(Array);
      expect(resourceContent.contents.length).toBe(1);
      const contentItem = resourceContent.contents[0];
      expect(contentItem.uri).toBe('weather://current');
      expect(contentItem.mimeType).toBe('application/json');
      expect(contentItem.text).toBeDefined();
      let parsedText: any = {};
      if (contentItem.text && typeof contentItem.text === 'string') {
        try {
          parsedText = JSON.parse(contentItem.text);
        } catch {
          // If parsing fails, parsedText remains an empty object
          // console.error("Failed to parse resource content text:", _e);
        }
      }
      expect(parsedText).toHaveProperty('location');
    });

    it('should subscribe and unsubscribe from a resource on a specific server', async () => {
      const serverName = 'weather';
      const resourceUri = 'weather://current';

      const subResult = await mcp.resources.subscribe(serverName, resourceUri);
      expect(subResult).toEqual({});

      const unsubResult = await mcp.resources.unsubscribe(serverName, resourceUri);
      expect(unsubResult).toEqual({});
    });

    it('should receive resource updated notification from a specific server', async () => {
      const serverName = 'weather';
      const resourceUri = 'weather://current';
      let notificationReceived = false;
      let receivedUri = '';

      await mcp.resources.list(); // Initial call to establish connection if needed
      // Create the promise for the notification BEFORE subscribing
      const resourceUpdatedPromise = new Promise<void>((resolve, reject) => {
        mcp.resources.onUpdated(serverName, (params: { uri: string }) => {
          if (params.uri === resourceUri) {
            notificationReceived = true;
            receivedUri = params.uri;
            resolve();
          } else {
            console.log(`[Test LOG] Received update for ${params.uri}, waiting for ${resourceUri}`);
          }
        });
        setTimeout(() => reject(new Error(`Timeout waiting for resourceUpdated notification for ${resourceUri}`)), 4500);
      });

      await mcp.resources.subscribe(serverName, resourceUri); // Ensure subscription is active

      await expect(resourceUpdatedPromise).resolves.toBeUndefined(); // Wait for the notification

      expect(notificationReceived).toBe(true);
      expect(receivedUri).toBe(resourceUri);

      await mcp.resources.unsubscribe(serverName, resourceUri); // Cleanup
    }, 5000);

    it('should receive resource list changed notification from a specific server', async () => {
      const serverName = 'weather';
      let notificationReceived = false;

      await mcp.resources.list(); // Initial call to establish connection

      const resourceListChangedPromise = new Promise<void>((resolve, reject) => {
        mcp.resources.onListChanged(serverName, () => {
          notificationReceived = true;
          resolve();
        });
        setTimeout(() => reject(new Error('Timeout waiting for resourceListChanged notification')), 4500);
      });

      // In a real scenario, something would trigger the server to send this.
      // For the test, we rely on the interval in weather.ts or a direct call if available.
      // Adding a small delay or an explicit trigger if the fixture supported it would be more robust.
      // For now, we assume the interval in weather.ts will eventually fire it.

      await expect(resourceListChangedPromise).resolves.toBeUndefined(); // Wait for the notification

      expect(notificationReceived).toBe(true);
    });

    it('should handle errors when getting resources', async () => {
      const errorClient = new MCPClient({
        id: 'error-test-client',
        servers: {
          weather: {
            url: new URL(`http://localhost:${weatherServerPort}/sse`),
          },
          nonexistentServer: {
            command: 'nonexistent-command',
            args: [],
          },
        },
      });

      try {
        const resources = await errorClient.resources.list();

        expect(resources).toHaveProperty('weather');
        expect(resources.weather).toBeDefined();
        expect(resources.weather.length).toBeGreaterThan(0);

        expect(resources).not.toHaveProperty('nonexistentServer');
      } finally {
        await errorClient.disconnect();
      }
    });
  })

  describe('Prompts', () => {
    it('should get prompts from connected MCP servers', async () => {
      const prompts = await mcp.prompts.list();

      expect(prompts).toHaveProperty('weather');
      expect(prompts['weather']).toBeDefined();
      expect(prompts['weather']).toHaveLength(3);

      // Verify that each expected resource exists with the correct structure
      const promptResources = prompts['weather'];
      const currentWeatherPrompt = promptResources.find(r => r.name === 'current');
      expect(currentWeatherPrompt).toBeDefined();
      expect(currentWeatherPrompt).toMatchObject({
        name: 'current',
        version: 'v1',
        description: expect.any(String),
        mimeType: 'application/json',
      });

      const forecast = promptResources.find(r => r.name === 'forecast');
      expect(forecast).toBeDefined();
      expect(forecast).toMatchObject({
        name: 'forecast',
        version: 'v1',
        description: expect.any(String),
        mimeType: 'application/json',
      });

      const historical = promptResources.find(r => r.name === 'historical');
      expect(historical).toBeDefined();
      expect(historical).toMatchObject({
        name: 'historical',
        version: 'v1',
        description: expect.any(String),
        mimeType: 'application/json',
      });
    });

    it('should get a specific prompt from a server', async () => {
      const {prompt, messages} = await mcp.prompts.get({serverName: 'weather', name: 'current'});
      expect(prompt).toBeDefined();
      expect(prompt).toMatchObject({
        name: 'current',
        version: 'v1',
        description: expect.any(String),
        mimeType: 'application/json',
      });
      expect(messages).toBeDefined();
      const messageItem = messages[0];
      let parsedText: any = {};
      if (messageItem.content.text && typeof messageItem.content.text === 'string') {
        try {
          parsedText = JSON.parse(messageItem.content.text);
        } catch {
          // If parsing fails, parsedText remains an empty object
          // console.error("Failed to parse resource content text:", _e);
        }
      }
      expect(parsedText).toHaveProperty('location');
    });

    it('should receive prompt list changed notification from a specific server', async () => {
      const serverName = 'weather';
      let notificationReceived = false;

      await mcp.prompts.list();

      const promptListChangedPromise = new Promise<void>((resolve, reject) => {
        mcp.prompts.onListChanged(serverName, () => {
          notificationReceived = true;
          resolve();
        });
        setTimeout(() => reject(new Error('Timeout waiting for promptListChanged notification')), 4500);
      });

      await expect(promptListChangedPromise).resolves.toBeUndefined();

      expect(notificationReceived).toBe(true);
    });

    it('should handle errors when getting prompts', async () => {
      const errorClient = new MCPClient({
        id: 'error-test-client',
        servers: {
          weather: {
            url: new URL(`http://localhost:${weatherServerPort}/sse`),
          },
          nonexistentServer: {
            command: 'nonexistent-command',
            args: [],
          },
        },
      });

      try {
        const prompts = await errorClient.prompts.list();

        expect(prompts).toHaveProperty('weather');
        expect(prompts['weather']).toBeDefined();
        expect(prompts['weather'].length).toBeGreaterThan(0);

        expect(prompts).not.toHaveProperty('nonexistentServer');
      } finally {
        await errorClient.disconnect();
      }
    });
  })

  describe('Instance Management', () => {
    it('should allow multiple instances with different IDs', async () => {
      const config2 = new MCPClient({
        id: 'custom-id',
        servers: {
          stockPrice: {
            command: 'npx',
            args: ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
            env: {
              FAKE_CREDS: 'test',
            },
          },
        },
      });

      expect(config2).not.toBe(mcp);
      await config2.disconnect();
    });

    it('should allow reuse of configuration after closing', async () => {
      await mcp.disconnect();

      const config2 = new MCPClient({
        servers: {
          stockPrice: {
            command: 'npx',
            args: ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
            env: {
              FAKE_CREDS: 'test',
            },
          },
          weather: {
            url: new URL(`http://localhost:${weatherServerPort}/sse`),
          },
        },
      });

      expect(config2).not.toBe(mcp);
      await config2.disconnect();
    });

    it('should throw error when creating duplicate instance without ID', async () => {
      const existingConfig = new MCPClient({
        servers: {
          stockPrice: {
            command: 'npx',
            args: ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
            env: {
              FAKE_CREDS: 'test',
            },
          },
        },
      });

      expect(
        () =>
          new MCPClient({
            servers: {
              stockPrice: {
                command: 'npx',
                args: ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
                env: {
                  FAKE_CREDS: 'test',
                },
              },
            },
          }),
      ).toThrow(/MCPClient was initialized multiple times/);

      await existingConfig.disconnect();
    });
  });
  describe('MCPClient Operation Timeouts', () => {
    it('should respect custom timeout in configuration', async () => {
      const config = new MCPClient({
        id: 'test-timeout-config',
        timeout: 3000, // 3 second timeout
        servers: {
          test: {
            command: 'node',
            args: [
              '-e',
              `
            const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
            const server = new Server({ name: 'test', version: '1.0.0' });
            setTimeout(() => process.exit(0), 2000); // 2 second delay
          `,
            ],
          },
        },
      });

      const error = await config.getTools().catch(e => e);
      expect(error).toBeDefined(); // Will throw since server exits before responding
      expect(error.message).not.toMatch(/Request timed out/);

      await config.disconnect();
    });

    it('should respect per-server timeout override', async () => {
      const config = new MCPClient({
        id: 'test-server-timeout-config',
        timeout: 500, // Global timeout of 500ms
        servers: {
          test: {
            command: 'node',
            args: [
              '-e',
              `
            const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
            const server = new Server({ name: 'test', version: '1.0.0' });
            setTimeout(() => process.exit(0), 2000); // 2 second delay
          `,
            ],
            timeout: 3000, // Server-specific timeout of 3s
          },
        },
      });

      // This should succeed since server timeout (3s) is longer than delay (2s)
      const error = await config.getTools().catch(e => e);
      expect(error).toBeDefined(); // Will throw since server exits before responding
      expect(error.message).not.toMatch(/Request timed out/);

      await config.disconnect();
    });
  });

  describe('MCPClient Connection Timeout', () => {
    it('should throw timeout error for slow starting server', async () => {
      const slowConfig = new MCPClient({
        id: 'test-slow-server',
        servers: {
          slowServer: {
            command: 'node',
            args: ['-e', 'setTimeout(() => process.exit(0), 65000)'], // Simulate a server that takes 65 seconds to start
            timeout: 1000,
          },
        },
      });

      await expect(slowConfig.getTools()).rejects.toThrow(/Request timed out/);
      await slowConfig.disconnect();
    });

    it('timeout should be longer than configured timeout', async () => {
      const slowConfig = new MCPClient({
        id: 'test-slow-server',
        timeout: 2000,
        servers: {
          slowServer: {
            command: 'node',
            args: ['-e', 'setTimeout(() => process.exit(0), 1000)'], // Simulate a server that takes 1 second to start
          },
        },
      });

      const error = await slowConfig.getTools().catch(e => e);
      expect(error).toBeDefined();
      expect(error.message).not.toMatch(/Request timed out/);
      await slowConfig.disconnect();
    });

    it('should respect per-server timeout configuration', async () => {
      const mixedConfig = new MCPClient({
        id: 'test-mixed-timeout',
        timeout: 1000, // Short global timeout
        servers: {
          quickServer: {
            command: 'node',
            args: ['-e', 'setTimeout(() => process.exit(0), 2000)'], // Takes 2 seconds to exit
          },
          slowServer: {
            command: 'node',
            args: ['-e', 'setTimeout(() => process.exit(0), 2000)'], // Takes 2 seconds to exit
            timeout: 3000, // But has a longer timeout
          },
        },
      });

      // Quick server should timeout
      await expect(mixedConfig.getTools()).rejects.toThrow(/Request timed out/);
      await mixedConfig.disconnect();
    });

    it('should handle connection errors gracefully', async () => {
      const badConfig = new MCPClient({
        servers: {
          badServer: {
            command: 'nonexistent-command',
            args: [],
          },
        },
      });

      await expect(badConfig.getTools()).rejects.toThrow();
      await badConfig.disconnect();
    });
  });

  describe('Schema Handling', () => {
    let complexClient: MCPClient;
    let mockLogHandler: LogHandler & ReturnType<typeof vi.fn>;

    beforeEach(async () => {
      mockLogHandler = vi.fn();

      complexClient = new MCPClient({
        id: 'complex-schema-test-client-log-handler-firecrawl',
        servers: {
          'firecrawl-mcp': {
            command: 'npx',
            args: ['-y', 'tsx', path.join(__dirname, '..', '__fixtures__/fire-crawl-complex-schema.ts')],
            logger: mockLogHandler,
          },
        },
      });
    });

    afterEach(async () => {
      mockLogHandler.mockClear();
      await complexClient?.disconnect().catch(() => { });
    });

    it('should process tools from firecrawl-mcp without crashing', async () => {
      const tools = await complexClient.getTools();

      Object.keys(allTools).forEach(toolName => {
        expect(tools).toHaveProperty(`${mcpServerName.replace(`-fixture`, ``)}_${toolName}`);
      });

      expect(mockLogHandler.mock.calls.length).toBeGreaterThan(0);
    });
  });

  describe('MCPClient Configuration', () => {
    let clientsToCleanup: MCPClient[] = [];

    afterEach(async () => {
      await Promise.all(
        clientsToCleanup.map(client =>
          client.disconnect().catch(e => console.error(`Error disconnecting client during test cleanup: ${e}`)),
        ),
      );
      clientsToCleanup = []; // Reset for the next test
    });

    it('should pass runtimeContext to the server logger function during tool execution', async () => {
      type TestContext = { channel: string; userId: string };
      const testContextInstance = new RuntimeContext<TestContext>();
      testContextInstance.set('channel', 'test-channel-123');
      testContextInstance.set('userId', 'user-abc-987');
      const loggerFn = vi.fn();

      const clientForTest = new MCPClient({
        servers: {
          stockPrice: {
            command: 'npx',
            args: ['tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
            env: { FAKE_CREDS: 'test' },
            logger: loggerFn,
          },
        },
      });
      clientsToCleanup.push(clientForTest);

      const tools = await clientForTest.getTools();
      const stockTool = tools['stockPrice_getStockPrice'];
      expect(stockTool).toBeDefined();

      await stockTool.execute({
        context: { symbol: 'MSFT' },
        runtimeContext: testContextInstance,
      });

      expect(loggerFn).toHaveBeenCalled();
      const callWithContext = loggerFn.mock.calls.find(call => {
        const logMessage = call[0] as LogMessage;
        return (
          logMessage.runtimeContext &&
          typeof logMessage.runtimeContext.get === 'function' &&
          logMessage.runtimeContext.get('channel') === 'test-channel-123' &&
          logMessage.runtimeContext.get('userId') === 'user-abc-987'
        );
      });
      expect(callWithContext).toBeDefined();
      const capturedLogMessage = callWithContext?.[0] as LogMessage;
      expect(capturedLogMessage?.serverName).toEqual('stockPrice');
    }, 15000);

    it('should pass runtimeContext to MCP logger when tool is called via an Agent', async () => {
      type TestAgentContext = { traceId: string; tenant: string };
      const agentTestContext = new RuntimeContext<TestAgentContext>();
      agentTestContext.set('traceId', 'agent-trace-xyz');
      agentTestContext.set('tenant', 'acme-corp');
      const loggerFn = vi.fn();

      const mcpClientForAgentTest = new MCPClient({
        id: 'mcp-for-agent-test-suite',
        servers: {
          stockPriceServer: {
            command: 'npx',
            args: ['tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
            env: { FAKE_CREDS: 'test' },
            logger: loggerFn,
          },
        },
      });
      clientsToCleanup.push(mcpClientForAgentTest);

      const agentName = 'stockAgentForContextTest';
      const agent = new Agent({
        name: agentName,
        model: openai('gpt-4o'),
        instructions: 'Use the getStockPrice tool to find the price of MSFT.',
        tools: await mcpClientForAgentTest.getTools(),
      });

      await agent.generate('What is the price of MSFT?', { runtimeContext: agentTestContext });

      expect(loggerFn).toHaveBeenCalled();
      const callWithAgentContext = loggerFn.mock.calls.find(call => {
        const logMessage = call[0] as LogMessage;
        return (
          logMessage.runtimeContext &&
          typeof logMessage.runtimeContext.get === 'function' &&
          logMessage.runtimeContext.get('traceId') === 'agent-trace-xyz' &&
          logMessage.runtimeContext.get('tenant') === 'acme-corp'
        );
      });
      expect(callWithAgentContext).toBeDefined();
      if (callWithAgentContext) {
        const capturedLogMessage = callWithAgentContext[0] as LogMessage;
        expect(capturedLogMessage?.serverName).toEqual('stockPriceServer');
      }
    }, 20000);

    it('should correctly use different runtimeContexts on sequential direct tool calls', async () => {
      const loggerFn = vi.fn();
      const clientForSeqTest = new MCPClient({
        id: 'mcp-sequential-context-test',
        servers: {
          stockPriceServer: {
            command: 'npx',
            args: ['tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')],
            env: { FAKE_CREDS: 'test' },
            logger: loggerFn,
          },
        },
      });
      clientsToCleanup.push(clientForSeqTest);

      const tools = await clientForSeqTest.getTools();
      const stockTool = tools['stockPriceServer_getStockPrice'];
      expect(stockTool).toBeDefined();

      type ContextA = { callId: string };
      const runtimeContextA = new RuntimeContext<ContextA>();
      runtimeContextA.set('callId', 'call-A-111');
      await stockTool.execute({ context: { symbol: 'MSFT' }, runtimeContext: runtimeContextA });

      expect(loggerFn).toHaveBeenCalled();
      let callsAfterA = [...loggerFn.mock.calls];
      const logCallForA = callsAfterA.find(
        call => (call[0] as LogMessage).runtimeContext?.get('callId') === 'call-A-111',
      );
      expect(logCallForA).toBeDefined();
      expect((logCallForA?.[0] as LogMessage)?.runtimeContext?.get('callId')).toBe('call-A-111');

      loggerFn.mockClear();

      type ContextB = { sessionId: string };
      const runtimeContextB = new RuntimeContext<ContextB>();
      runtimeContextB.set('sessionId', 'session-B-222');
      await stockTool.execute({ context: { symbol: 'GOOG' }, runtimeContext: runtimeContextB });

      expect(loggerFn).toHaveBeenCalled();
      let callsAfterB = [...loggerFn.mock.calls];
      const logCallForB = callsAfterB.find(
        call => (call[0] as LogMessage).runtimeContext?.get('sessionId') === 'session-B-222',
      );
      expect(logCallForB).toBeDefined();
      expect((logCallForB?.[0] as LogMessage)?.runtimeContext?.get('sessionId')).toBe('session-B-222');

      const contextALeak = callsAfterB.some(
        call => (call[0] as LogMessage).runtimeContext?.get('callId') === 'call-A-111',
      );
      expect(contextALeak).toBe(false);
    }, 20000);

    it('should isolate runtimeContext between different servers on the same MCPClient', async () => {
      const sharedLoggerFn = vi.fn();

      const clientWithTwoServers = new MCPClient({
        id: 'mcp-multi-server-context-isolation',
        servers: {
          serverX: {
            command: 'npx',
            args: ['tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')], // Re-use fixture, tool name will differ by server
            logger: sharedLoggerFn,
            env: { FAKE_CREDS: 'serverX-creds' }, // Make env slightly different for clarity if needed
          },
          serverY: {
            command: 'npx',
            args: ['tsx', path.join(__dirname, '..', '__fixtures__/stock-price.ts')], // Re-use fixture
            logger: sharedLoggerFn,
            env: { FAKE_CREDS: 'serverY-creds' },
          },
        },
      });
      clientsToCleanup.push(clientWithTwoServers);

      const tools = await clientWithTwoServers.getTools();
      const toolX = tools['serverX_getStockPrice'];
      const toolY = tools['serverY_getStockPrice'];
      expect(toolX).toBeDefined();
      expect(toolY).toBeDefined();

      // --- Call tool on Server X with contextX ---
      type ContextX = { requestId: string };
      const runtimeContextX = new RuntimeContext<ContextX>();
      runtimeContextX.set('requestId', 'req-X-001');

      await toolX.execute({ context: { symbol: 'AAA' }, runtimeContext: runtimeContextX });

      expect(sharedLoggerFn).toHaveBeenCalled();
      let callsAfterToolX = [...sharedLoggerFn.mock.calls];
      const logCallForX = callsAfterToolX.find(call => {
        const logMessage = call[0] as LogMessage;
        return logMessage.serverName === 'serverX' && logMessage.runtimeContext?.get('requestId') === 'req-X-001';
      });
      expect(logCallForX).toBeDefined();
      expect((logCallForX?.[0] as LogMessage)?.runtimeContext?.get('requestId')).toBe('req-X-001');

      sharedLoggerFn.mockClear(); // Clear for next distinct operation

      // --- Call tool on Server Y with contextY ---
      type ContextY = { customerId: string };
      const runtimeContextY = new RuntimeContext<ContextY>();
      runtimeContextY.set('customerId', 'cust-Y-002');

      await toolY.execute({ context: { symbol: 'BBB' }, runtimeContext: runtimeContextY });

      expect(sharedLoggerFn).toHaveBeenCalled();
      let callsAfterToolY = [...sharedLoggerFn.mock.calls];
      const logCallForY = callsAfterToolY.find(call => {
        const logMessage = call[0] as LogMessage;
        return logMessage.serverName === 'serverY' && logMessage.runtimeContext?.get('customerId') === 'cust-Y-002';
      });
      expect(logCallForY).toBeDefined();
      expect((logCallForY?.[0] as LogMessage)?.runtimeContext?.get('customerId')).toBe('cust-Y-002');

      // Ensure contextX did not leak into logs from serverY's operation
      const contextXLeakInYLogs = callsAfterToolY.some(call => {
        const logMessage = call[0] as LogMessage;
        return logMessage.runtimeContext?.get('requestId') === 'req-X-001';
      });
      expect(contextXLeakInYLogs).toBe(false);
    }, 25000); // Increased timeout for multiple server ops
  });
});
