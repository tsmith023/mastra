import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { join } from 'path';
import process from 'process';
import { Deployer } from '@mastra/deployer';

interface EnvVar {
  key: string;
  value: string;
  target: ('production' | 'preview' | 'development')[];
  type: 'plain' | 'secret';
}

interface VercelError {
  message: string;
  code: string;
}

export class VercelDeployer extends Deployer {
  private teamSlug: string;
  private projectName: string;
  private token: string;

  constructor({ teamSlug, projectName, token }: { teamSlug: string; projectName: string; token: string }) {
    super({ name: 'VERCEL' });

    this.teamSlug = teamSlug;
    this.projectName = projectName;
    this.token = token;
  }

  private getProjectId({ dir }: { dir: string }): string {
    const projectJsonPath = join(dir, 'output', '.vercel', 'project.json');

    try {
      const projectJson = JSON.parse(readFileSync(projectJsonPath, 'utf-8'));
      return projectJson.projectId;
    } catch {
      throw new Error('Could not find project ID. Make sure the project has been deployed first.');
    }
  }

  private async getTeamId(): Promise<string> {
    const response = await fetch(`https://api.vercel.com/v2/teams`, {
      headers: {
        Authorization: `Bearer ${this.token}`,
      },
    });

    const res = (await response.json()) as any;
    const teams = res.teams;
    return teams.find((team: any) => team.slug === this.teamSlug)?.id;
  }

  private async syncEnv(envVars: Map<string, string>, { outputDirectory }: { outputDirectory: string }) {
    console.log('Syncing environment variables...');

    // Transform env vars into the format expected by Vercel API
    const vercelEnvVars: EnvVar[] = Array.from(envVars.entries()).map(([key, value]) => {
      if (!key || !value) {
        throw new Error(`Invalid environment variable format: ${key || value}`);
      }

      return {
        key,
        value,
        target: ['production', 'preview', 'development'],
        type: 'plain',
      };
    });

    try {
      const projectId = this.getProjectId({ dir: outputDirectory });
      const teamId = await this.getTeamId();

      const response = await fetch(
        `https://api.vercel.com/v10/projects/${projectId}/env?teamId=${teamId}&upsert=true`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${this.token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(vercelEnvVars),
        },
      );

      if (!response.ok) {
        const error = (await response.json()) as VercelError;
        throw new Error(`Failed to sync environment variables: ${error.message}`);
      }

      console.log('✓ Successfully synced environment variables');
    } catch (error) {
      if (error instanceof Error) {
        console.error('Failed to sync environment variables:', error.message);
      } else {
        console.error('Failed to sync environment variables:', error);
      }
      throw error;
    }
  }

  async prepare(outputDirectory: string): Promise<void> {
    await super.prepare(outputDirectory);
  }

  private getEntry(): string {
    return `
import { handle } from 'hono/vercel'
import { mastra } from '#mastra';
import { createHonoServer } from '#server';
import { evaluate } from '@mastra/core/eval';
import { AvailableHooks, registerHook } from '@mastra/core/hooks';
import { TABLE_EVALS } from '@mastra/core/storage';
import { checkEvalStorageFields } from '@mastra/core/utils';

registerHook(AvailableHooks.ON_GENERATION, ({ input, output, metric, runId, agentName, instructions }) => {
  evaluate({
    agentName,
    input,
    metric,
    output,
    runId,
    globalRunId: runId,
    instructions,
  });
});

registerHook(AvailableHooks.ON_EVALUATION, async traceObject => {
  const storage = mastra.getStorage();
  if (storage) {
    // Check for required fields
    const logger = mastra?.getLogger();
    const areFieldsValid = checkEvalStorageFields(traceObject, logger);
    if (!areFieldsValid) return;

    await storage.insert({
      tableName: TABLE_EVALS,
      record: {
        input: traceObject.input,
        output: traceObject.output,
        result: JSON.stringify(traceObject.result || {}),
        agent_name: traceObject.agentName,
        metric_name: traceObject.metricName,
        instructions: traceObject.instructions,
        test_info: null,
        global_run_id: traceObject.globalRunId,
        run_id: traceObject.runId,
        created_at: new Date().toISOString(),
      },
    });
  }
});

const app = await createHonoServer(mastra);

export const GET = handle(app);
export const POST = handle(app);
export const PUT = handle(app);
export const DELETE = handle(app);
export const OPTIONS = handle(app);
export const HEAD = handle(app);
`;
  }

  private writeVercelJSON(outputDirectory: string, files: string[] = ['./*']) {
    writeFileSync(
      join(outputDirectory, this.outputDir, 'vercel.json'),
      JSON.stringify(
        {
          version: 2,
          installCommand: 'npm install --omit=dev',
          builds: [
            {
              src: 'index.mjs',
              use: '@vercel/node',
              config: { includeFiles: files },
            },
          ],
          routes: [
            {
              src: '/(.*)',
              dest: 'index.mjs',
            },
          ],
        },
        null,
        2,
      ),
    );
  }

  async bundle(entryFile: string, outputDirectory: string, toolsPaths: string[]): Promise<void> {
    const result = await this._bundle(this.getEntry(), entryFile, outputDirectory, toolsPaths);

    // read dist files one level deep in the output directory
    const files = readdirSync(join(outputDirectory, this.outputDir), {
      recursive: true,
    });

    const filesWithoutNodeModules = files.filter(
      file => typeof file === 'string' && !file.startsWith('node_modules'),
    ) as string[];

    this.writeVercelJSON(outputDirectory, filesWithoutNodeModules);

    return result;
  }

  async deploy(): Promise<void> {
    this.logger?.info('Deploying to Vercel failed. Please use the Vercel dashboard to deploy.');
  }

  async lint(entryFile: string, outputDirectory: string, toolsPaths: string[]): Promise<void> {
    await super.lint(entryFile, outputDirectory, toolsPaths);

    await super.lint(entryFile, outputDirectory, toolsPaths);

    const hasLibsql = (await this.deps.checkDependencies(['@mastra/libsql'])) === `ok`;

    if (hasLibsql) {
      this.logger.error(
        `Vercel Deployer does not support @libsql/client(which may have been installed by @mastra/libsql) as a dependency. 
        Use other Mastra Storage options instead e.g @mastra/pg`,
      );
      process.exit(1);
    }
  }
}
