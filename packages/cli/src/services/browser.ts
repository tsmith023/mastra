/**
 * The following is modified based on source found in
 * https://github.com/vitejs/vite/tree/main
 *
 * MIT Licensed
 * Copyright (c) 2019-present, Vite.js
 * https://github.com/vitejs/vite/blob/main/LICENSE
 */

import { exec } from 'node:child_process';
import type { ExecOptions } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import open from 'open';
import type { Options } from 'open';

/**
 * Reads the BROWSER environment variable and decides what to do with it.
 */
export function openBrowser(url: string, opt: string | true): void {
  // The browser executable to open.
  // See https://github.com/sindresorhus/open#app for documentation.
  const browser = typeof opt === 'string' ? opt : process.env.BROWSER || '';
  if (browser.toLowerCase() !== 'none') {
    const browserArgs = process.env.BROWSER_ARGS ? process.env.BROWSER_ARGS.split(' ') : [];
    void startBrowserProcess(browser, browserArgs, url);
  }
}

const supportedChromiumBrowsers = [
  'Google Chrome Canary',
  'Google Chrome Dev',
  'Google Chrome Beta',
  'Google Chrome',
  'Microsoft Edge',
  'Brave Browser',
  'Vivaldi',
  'Chromium',
  'Arc',
];

async function startBrowserProcess(browser: string | undefined, browserArgs: string[], url: string) {
  // If we're on OS X, the user hasn't specifically
  // requested a different browser, we can try opening
  // a Chromium browser with AppleScript. This lets us reuse an
  // existing tab when possible instead of creating a new one.
  const preferredOSXBrowser = browser === 'google chrome' ? 'Google Chrome' : browser;
  const shouldTryOpenChromeWithAppleScript =
    process.platform === 'darwin' && (!preferredOSXBrowser || supportedChromiumBrowsers.includes(preferredOSXBrowser));

  if (shouldTryOpenChromeWithAppleScript) {
    try {
      const ps = await execAsync('ps cax');
      const openedBrowser =
        preferredOSXBrowser && ps.includes(preferredOSXBrowser)
          ? preferredOSXBrowser
          : supportedChromiumBrowsers.find(b => ps.includes(b));
      if (openedBrowser) {
        const packageDir = dirname(fileURLToPath(import.meta.resolve('mastra/package.json')));

        // Try our best to reuse existing tab with AppleScript
        await execAsync(`osascript openChrome.applescript "${url}" "${openedBrowser}"`, {
          cwd: join(packageDir, 'bin'),
        });
        return true;
      }
    } catch {
      // Ignore errors
    }
  }

  // Another special case: on OS X, check if BROWSER has been set to "open".
  // In this case, instead of passing the string `open` to `open` function (which won't work),
  // just ignore it (thus ensuring the intended behavior, i.e. opening the system browser):
  // https://github.com/facebook/create-react-app/pull/1690#issuecomment-283518768
  if (process.platform === 'darwin' && browser === 'open') {
    browser = undefined;
  }

  // Fallback to open
  // (It will always open new tab)
  try {
    const options: Options = browser ? { app: { name: browser, arguments: browserArgs } } : {};

    new Promise((_, reject) => {
      open(url, options)
        .then(subprocess => {
          subprocess.on('error', reject);
        })
        .catch(reject);
    }).catch(err => {
      console.error(err.stack || err.message);
    });

    return true;
  } catch {
    return false;
  }
}

function execAsync(command: string, options?: ExecOptions): Promise<string> {
  return new Promise((resolve, reject) => {
    exec(command, options, (error, stdout) => {
      if (error) {
        reject(error);
      } else {
        resolve(stdout.toString());
      }
    });
  });
}
