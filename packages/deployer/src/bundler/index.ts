import { existsSync } from 'node:fs';
import { stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MastraBundler } from '@mastra/core/bundler';
import virtual from '@rollup/plugin-virtual';
import fsExtra, { copy, ensureDir, readJSON, emptyDir } from 'fs-extra/esm';
import { globby } from 'globby';
import resolveFrom from 'resolve-from';
import type { InputOptions, OutputOptions } from 'rollup';

import { analyzeBundle } from '../build/analyze';
import { createBundler as createBundlerUtil, getInputOptions } from '../build/bundler';
import { writeTelemetryConfig } from '../build/telemetry';
import { DepsService } from '../services/deps';
import { FileService } from '../services/fs';
import {
  collectTransitiveWorkspaceDependencies,
  createWorkspacePackageMap,
  packWorkspaceDependencies,
} from './workspaceDependencies';

export abstract class Bundler extends MastraBundler {
  protected analyzeOutputDir = '.build';
  protected outputDir = 'output';

  constructor(name: string, component: 'BUNDLER' | 'DEPLOYER' = 'BUNDLER') {
    super({ name, component });
  }

  async prepare(outputDirectory: string): Promise<void> {
    // Clean up the output directory first
    await emptyDir(outputDirectory);

    await ensureDir(join(outputDirectory, this.analyzeOutputDir));
    await ensureDir(join(outputDirectory, this.outputDir));
  }

  async writeInstrumentationFile(outputDirectory: string) {
    const instrumentationFile = join(outputDirectory, 'instrumentation.mjs');
    const __dirname = dirname(fileURLToPath(import.meta.url));

    await copy(join(__dirname, 'templates', 'instrumentation-template.js'), instrumentationFile);
  }

  async writePackageJson(
    outputDirectory: string,
    dependencies: Map<string, string>,
    resolutions?: Record<string, string>,
  ) {
    this.logger.debug(`Writing project's package.json`);
    await ensureDir(outputDirectory);
    const pkgPath = join(outputDirectory, 'package.json');

    const dependenciesMap = new Map();
    for (const [key, value] of dependencies.entries()) {
      if (key.startsWith('@')) {
        // Handle scoped packages (e.g. @org/package)
        const pkgChunks = key.split('/');
        dependenciesMap.set(`${pkgChunks[0]}/${pkgChunks[1]}`, value);
      } else {
        // For non-scoped packages, take only the first part before any slash
        const pkgName = key.split('/')[0] || key;
        dependenciesMap.set(pkgName, value);
      }
    }

    // add telemetry dependencies
    dependenciesMap.set('@opentelemetry/core', '^2.0.1');
    dependenciesMap.set('@opentelemetry/auto-instrumentations-node', '^0.59.0');
    dependenciesMap.set('@opentelemetry/exporter-trace-otlp-grpc', '^0.201.0');
    dependenciesMap.set('@opentelemetry/exporter-trace-otlp-http', '^0.201.0');
    dependenciesMap.set('@opentelemetry/resources', '^2.0.1');
    dependenciesMap.set('@opentelemetry/sdk-node', '^0.201.0');
    dependenciesMap.set('@opentelemetry/sdk-trace-base', '^2.0.1');
    dependenciesMap.set('@opentelemetry/semantic-conventions', '^1.33.0');
    dependenciesMap.set('@opentelemetry/instrumentation', '^0.202.0');

    await writeFile(
      pkgPath,
      JSON.stringify(
        {
          name: 'server',
          version: '1.0.0',
          description: '',
          type: 'module',
          main: 'index.mjs',
          scripts: {
            start: 'node --import=./instrumentation.mjs --import=@opentelemetry/instrumentation/hook.mjs ./index.mjs',
          },
          author: 'Mastra',
          license: 'ISC',
          dependencies: Object.fromEntries(dependenciesMap.entries()),
          ...(Object.keys(resolutions ?? {}).length > 0 && { resolutions }),
          pnpm: {
            neverBuiltDependencies: [],
          },
        },
        null,
        2,
      ),
    );
  }

  protected createBundler(inputOptions: InputOptions, outputOptions: Partial<OutputOptions> & { dir: string }) {
    return createBundlerUtil(inputOptions, outputOptions);
  }

  protected async analyze(entry: string, mastraFile: string, outputDirectory: string) {
    return await analyzeBundle(entry, mastraFile, join(outputDirectory, this.analyzeOutputDir), 'node', this.logger);
  }

  protected async installDependencies(outputDirectory: string, rootDir = process.cwd()) {
    const deps = new DepsService(rootDir);
    deps.__setLogger(this.logger);

    await deps.install({ dir: join(outputDirectory, this.outputDir) });
  }

  protected async copyPublic(mastraDir: string, outputDirectory: string) {
    const publicDir = join(mastraDir, 'public');

    try {
      await stat(publicDir);
    } catch {
      return;
    }

    await copy(publicDir, join(outputDirectory, this.outputDir));
  }

  protected async getBundlerOptions(
    serverFile: string,
    mastraEntryFile: string,
    analyzedBundleInfo: Awaited<ReturnType<typeof analyzeBundle>>,
    toolsPaths: string[],
  ) {
    const inputOptions: InputOptions = await getInputOptions(mastraEntryFile, analyzedBundleInfo, 'node', {
      'process.env.NODE_ENV': JSON.stringify('production'),
    });
    const isVirtual = serverFile.includes('\n') || existsSync(serverFile);

    const toolsInputOptions = await this.getToolsInputOptions(toolsPaths);

    if (isVirtual) {
      inputOptions.input = { index: '#entry', ...toolsInputOptions };

      if (Array.isArray(inputOptions.plugins)) {
        inputOptions.plugins.unshift(virtual({ '#entry': serverFile }));
      } else {
        inputOptions.plugins = [virtual({ '#entry': serverFile })];
      }
    } else {
      inputOptions.input = { index: serverFile, ...toolsInputOptions };
    }

    return inputOptions;
  }

  async getToolsInputOptions(toolsPaths: string[]) {
    const inputs: Record<string, string> = {};

    for (const toolPath of toolsPaths) {
      const expandedPaths = await globby(toolPath, {});

      for (const path of expandedPaths) {
        if (await fsExtra.pathExists(path)) {
          const fileService = new FileService();
          const entryFile = fileService.getFirstExistingFile([
            join(path, 'index.ts'),
            join(path, 'index.js'),
            path, // if path itself is a file
          ]);

          // if it doesn't exist or is a dir skip it. using a dir as a tool will crash the process
          if (!entryFile || (await stat(entryFile)).isDirectory()) {
            this.logger.warn(`No entry file found in ${path}, skipping...`);
            continue;
          }

          const uniqueToolID = crypto.randomUUID();
          inputs[`tools/${uniqueToolID}`] = entryFile;
        } else {
          this.logger.warn(`Tool path ${path} does not exist, skipping...`);
        }
      }
    }

    return inputs;
  }

  protected async _bundle(
    serverFile: string,
    mastraEntryFile: string,
    outputDirectory: string,
    toolsPaths: string[] = [],
    bundleLocation: string = join(outputDirectory, this.outputDir),
  ): Promise<void> {
    this.logger.info('Start bundling Mastra');

    const analyzedBundleInfo = await analyzeBundle(
      serverFile,
      mastraEntryFile,
      join(outputDirectory, this.analyzeOutputDir),
      'node',
      this.logger,
    );

    const { externalDependencies } = await writeTelemetryConfig(mastraEntryFile, join(outputDirectory, this.outputDir));

    const dependenciesToInstall = new Map<string, string>();
    // Add extenal dependencies from telemetry file
    for (const external of externalDependencies) {
      dependenciesToInstall.set(external, 'latest');
    }

    const workspaceMap = await createWorkspacePackageMap();
    const workspaceDependencies = new Set<string>();
    for (const dep of analyzedBundleInfo.externalDependencies) {
      try {
        const pkgPath = resolveFrom(mastraEntryFile, `${dep}/package.json`);
        const pkg = await readJSON(pkgPath);

        if (workspaceMap.has(pkg.name)) {
          workspaceDependencies.add(pkg.name);
          continue;
        }

        dependenciesToInstall.set(dep, pkg.version);
      } catch {
        dependenciesToInstall.set(dep, 'latest');
      }
    }

    let resolutions: Record<string, string> = {};
    if (workspaceDependencies.size > 0) {
      try {
        const result = collectTransitiveWorkspaceDependencies({
          workspaceMap,
          initialDependencies: workspaceDependencies,
          logger: this.logger,
        });
        resolutions = result.resolutions;

        // Update dependenciesToInstall with the resolved TGZ paths
        Object.entries(resolutions).forEach(([pkgName, tgzPath]) => {
          dependenciesToInstall.set(pkgName, tgzPath);
        });

        await packWorkspaceDependencies({
          workspaceMap,
          usedWorkspacePackages: result.usedWorkspacePackages,
          bundleOutputDir: join(outputDirectory, this.outputDir),
          logger: this.logger,
        });
      } catch (error) {
        this.logger.error(`Failed to collect workspace dependencies: ${error}`);
        return;
      }
    }

    await this.writePackageJson(join(outputDirectory, this.outputDir), dependenciesToInstall, resolutions);
    await this.writeInstrumentationFile(join(outputDirectory, this.outputDir));

    this.logger.info('Bundling Mastra application');
    const inputOptions: InputOptions = await this.getBundlerOptions(
      serverFile,
      mastraEntryFile,
      analyzedBundleInfo,
      toolsPaths,
    );
    const bundler = await this.createBundler(inputOptions, {
      dir: bundleLocation,
      manualChunks: {
        mastra: ['#mastra'],
      },
    });

    await bundler.write();
    const toolsInputOptions = Array.from(Object.keys(inputOptions.input || {}))
      .filter(key => key.startsWith('tools/'))
      .map(key => `./${key}.mjs`);

    await writeFile(join(bundleLocation, 'tools.mjs'), `export const tools = ${JSON.stringify(toolsInputOptions)};`);
    this.logger.info('Bundling Mastra done');

    this.logger.info('Copying public files');
    await this.copyPublic(dirname(mastraEntryFile), outputDirectory);
    this.logger.info('Done copying public files');

    this.logger.info('Installing dependencies');
    await this.installDependencies(outputDirectory);

    this.logger.info('Done installing dependencies');
  }

  async lint(_entryFile: string, _outputDirectory: string, toolsPaths: string[]): Promise<void> {
    const toolsInputOptions = await this.getToolsInputOptions(toolsPaths);
    const toolsLength = Object.keys(toolsInputOptions).length;
    if (toolsLength > 0) {
      this.logger.info(`Found ${toolsLength} ${toolsLength === 1 ? 'tool' : 'tools'}`);
    }
  }
}
