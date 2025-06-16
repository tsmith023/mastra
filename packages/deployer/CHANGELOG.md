# @mastra/deployer

## 0.10.6-alpha.0

### Patch Changes

- 2d12edd: dependencies updates:
  - Updated dependency [`rollup@^4.43.0` ↗︎](https://www.npmjs.com/package/rollup/v/4.43.0) (from `^4.42.0`, in `dependencies`)
- 63f6b7d: dependencies updates:
  - Updated dependency [`detect-libc@^2.0.4` ↗︎](https://www.npmjs.com/package/detect-libc/v/2.0.4) (from `^2.0.3`, in `dependencies`)
  - Updated dependency [`esbuild@^0.25.5` ↗︎](https://www.npmjs.com/package/esbuild/v/0.25.5) (from `^0.25.1`, in `dependencies`)
  - Updated dependency [`rollup@^4.42.0` ↗︎](https://www.npmjs.com/package/rollup/v/4.42.0) (from `^4.41.1`, in `dependencies`)
  - Updated dependency [`zod@^3.25.57` ↗︎](https://www.npmjs.com/package/zod/v/3.25.57) (from `^3.25.56`, in `dependencies`)
- 36f1c36: MCP Client and Server streamable http fixes
- 53d3c37: Get workflows from an agent if not found from Mastra instance #5083
- Updated dependencies [63f6b7d]
- Updated dependencies [36f1c36]
- Updated dependencies [10d352e]
- Updated dependencies [53d3c37]
  - @mastra/core@0.10.6-alpha.0
  - @mastra/server@0.10.6-alpha.0

## 0.10.5

### Patch Changes

- 8725d02: Remove swaggerUI and openAPI url when server starts
- 105f872: Fix body already in use for POST requests
- Updated dependencies [1ba421d]
- Updated dependencies [13c97f9]
  - @mastra/server@0.10.5
  - @mastra/core@0.10.5

## 0.10.4

### Patch Changes

- d1ed912: dependencies updates:
  - Updated dependency [`dotenv@^16.5.0` ↗︎](https://www.npmjs.com/package/dotenv/v/16.5.0) (from `^16.4.7`, in `dependencies`)
- f595975: dependencies updates:
  - Updated dependency [`rollup@^4.41.1` ↗︎](https://www.npmjs.com/package/rollup/v/4.41.1) (from `^4.35.0`, in `dependencies`)
- d90c49f: dependencies updates:
  - Updated dependency [`@babel/core@^7.27.4` ↗︎](https://www.npmjs.com/package/@babel/core/v/7.27.4) (from `^7.26.10`, in `dependencies`)
  - Updated dependency [`@babel/helper-module-imports@^7.27.1` ↗︎](https://www.npmjs.com/package/@babel/helper-module-imports/v/7.27.1) (from `^7.25.9`, in `dependencies`)
  - Updated dependency [`@rollup/plugin-node-resolve@^16.0.1` ↗︎](https://www.npmjs.com/package/@rollup/plugin-node-resolve/v/16.0.1) (from `^16.0.0`, in `dependencies`)
  - Updated dependency [`hono@^4.7.11` ↗︎](https://www.npmjs.com/package/hono/v/4.7.11) (from `^4.7.4`, in `dependencies`)
- 1ccccff: dependencies updates:
  - Updated dependency [`zod@^3.25.56` ↗︎](https://www.npmjs.com/package/zod/v/3.25.56) (from `^3.24.3`, in `dependencies`)
- 1ccccff: dependencies updates:
  - Updated dependency [`zod@^3.25.56` ↗︎](https://www.npmjs.com/package/zod/v/3.25.56) (from `^3.24.3`, in `dependencies`)
- afd9fda: Reset retry-count on code change and only retry if server actually is running

  Fixes #4563

- f1f1f1b: Add basic filtering capabilities to logs
- 9597ee5: Hoist runtimeContext from POST request into middleware
- 82090c1: Add pagination to logs
- 69f6101: Add reason to tools import error on server start
- 514fdde: Move opentelemetry deps to mastra output to remove @mastra/core dependency
- bebd27c: Only apply <placeholder> text inside instructions in the playground ui
- Updated dependencies [d1ed912]
- Updated dependencies [f6fd25f]
- Updated dependencies [dffb67b]
- Updated dependencies [f1f1f1b]
- Updated dependencies [925ab94]
- Updated dependencies [9597ee5]
- Updated dependencies [f9816ae]
- Updated dependencies [82090c1]
- Updated dependencies [1b443fd]
- Updated dependencies [ce97900]
- Updated dependencies [f1309d3]
- Updated dependencies [bebd27c]
- Updated dependencies [14a2566]
- Updated dependencies [f7f8293]
- Updated dependencies [48eddb9]
  - @mastra/core@0.10.4
  - @mastra/server@0.10.4

## 0.10.4-alpha.3

### Patch Changes

- Updated dependencies [925ab94]
  - @mastra/core@0.10.4-alpha.3
  - @mastra/server@0.10.4-alpha.3

## 0.10.4-alpha.2

### Patch Changes

- Updated dependencies [48eddb9]
  - @mastra/core@0.10.4-alpha.2
  - @mastra/server@0.10.4-alpha.2

## 0.10.4-alpha.1

### Patch Changes

- d90c49f: dependencies updates:
  - Updated dependency [`@babel/core@^7.27.4` ↗︎](https://www.npmjs.com/package/@babel/core/v/7.27.4) (from `^7.26.10`, in `dependencies`)
  - Updated dependency [`@babel/helper-module-imports@^7.27.1` ↗︎](https://www.npmjs.com/package/@babel/helper-module-imports/v/7.27.1) (from `^7.25.9`, in `dependencies`)
  - Updated dependency [`@rollup/plugin-node-resolve@^16.0.1` ↗︎](https://www.npmjs.com/package/@rollup/plugin-node-resolve/v/16.0.1) (from `^16.0.0`, in `dependencies`)
  - Updated dependency [`hono@^4.7.11` ↗︎](https://www.npmjs.com/package/hono/v/4.7.11) (from `^4.7.4`, in `dependencies`)
- 1ccccff: dependencies updates:
  - Updated dependency [`zod@^3.25.56` ↗︎](https://www.npmjs.com/package/zod/v/3.25.56) (from `^3.24.3`, in `dependencies`)
- 1ccccff: dependencies updates:
  - Updated dependency [`zod@^3.25.56` ↗︎](https://www.npmjs.com/package/zod/v/3.25.56) (from `^3.24.3`, in `dependencies`)
- 9597ee5: Hoist runtimeContext from POST request into middleware
- 514fdde: Move opentelemetry deps to mastra output to remove @mastra/core dependency
- bebd27c: Only apply <placeholder> text inside instructions in the playground ui
- Updated dependencies [f6fd25f]
- Updated dependencies [dffb67b]
- Updated dependencies [9597ee5]
- Updated dependencies [f1309d3]
- Updated dependencies [bebd27c]
- Updated dependencies [f7f8293]
  - @mastra/core@0.10.4-alpha.1
  - @mastra/server@0.10.4-alpha.1

## 0.10.4-alpha.0

### Patch Changes

- d1ed912: dependencies updates:
  - Updated dependency [`dotenv@^16.5.0` ↗︎](https://www.npmjs.com/package/dotenv/v/16.5.0) (from `^16.4.7`, in `dependencies`)
- f595975: dependencies updates:
  - Updated dependency [`rollup@^4.41.1` ↗︎](https://www.npmjs.com/package/rollup/v/4.41.1) (from `^4.35.0`, in `dependencies`)
- afd9fda: Reset retry-count on code change and only retry if server actually is running

  Fixes #4563

- f1f1f1b: Add basic filtering capabilities to logs
- 82090c1: Add pagination to logs
- 69f6101: Add reason to tools import error on server start
- Updated dependencies [d1ed912]
- Updated dependencies [f1f1f1b]
- Updated dependencies [f9816ae]
- Updated dependencies [82090c1]
- Updated dependencies [1b443fd]
- Updated dependencies [ce97900]
- Updated dependencies [14a2566]
  - @mastra/core@0.10.4-alpha.0
  - @mastra/server@0.10.4-alpha.0

## 0.10.3

### Patch Changes

- Updated dependencies [2b0fc7e]
  - @mastra/core@0.10.3
  - @mastra/server@0.10.3

## 0.10.3-alpha.0

### Patch Changes

- Updated dependencies [2b0fc7e]
  - @mastra/core@0.10.3-alpha.0
  - @mastra/server@0.10.3-alpha.0

## 0.10.2

### Patch Changes

- e8d2aff: Fix non-scoped packages in mastra build
- f73e11b: fix telemetry disabled not working on playground
- 1fcc048: chore: generate sourcemaps in dev build
- f946acf: Filter out dynamic imports by node builtins
- add596e: Mastra protected auth
- ecebbeb: Mastra core auth abstract definition
- 4187ed4: Fix mcp server api openapijson
- f0d559f: Fix peerdeps for alpha channel
- Updated dependencies [ee77e78]
- Updated dependencies [592a2db]
- Updated dependencies [e5dc18d]
- Updated dependencies [ab5adbe]
- Updated dependencies [1e8bb40]
- Updated dependencies [1b5fc55]
- Updated dependencies [195c428]
- Updated dependencies [f73e11b]
- Updated dependencies [37643b8]
- Updated dependencies [e2228f6]
- Updated dependencies [99fd6cf]
- Updated dependencies [a399086]
- Updated dependencies [c5bf1ce]
- Updated dependencies [add596e]
- Updated dependencies [8dc94d8]
- Updated dependencies [ecebbeb]
- Updated dependencies [79d5145]
- Updated dependencies [422ee9e]
- Updated dependencies [12b7002]
- Updated dependencies [f0d559f]
- Updated dependencies [2901125]
- Updated dependencies [a0ebc3f]
  - @mastra/core@0.10.2
  - @mastra/server@0.10.2

## 0.10.2-alpha.8

### Patch Changes

- Updated dependencies [37643b8]
- Updated dependencies [79d5145]
  - @mastra/core@0.10.2-alpha.8
  - @mastra/server@0.10.2-alpha.8

## 0.10.2-alpha.7

### Patch Changes

- Updated dependencies [a399086]
  - @mastra/server@0.10.2-alpha.7
  - @mastra/core@0.10.2-alpha.7

## 0.10.2-alpha.6

### Patch Changes

- 1fcc048: chore: generate sourcemaps in dev build
- Updated dependencies [99fd6cf]
- Updated dependencies [8dc94d8]
  - @mastra/core@0.10.2-alpha.6
  - @mastra/server@0.10.2-alpha.6

## 0.10.2-alpha.5

### Patch Changes

- add596e: Mastra protected auth
- ecebbeb: Mastra core auth abstract definition
- Updated dependencies [1b5fc55]
- Updated dependencies [add596e]
- Updated dependencies [ecebbeb]
  - @mastra/server@0.10.2-alpha.5
  - @mastra/core@0.10.2-alpha.5

## 0.10.2-alpha.4

### Patch Changes

- Updated dependencies [c5bf1ce]
- Updated dependencies [12b7002]
  - @mastra/server@0.10.2-alpha.4
  - @mastra/core@0.10.2-alpha.4

## 0.10.2-alpha.3

### Patch Changes

- f73e11b: fix telemetry disabled not working on playground
- f946acf: Filter out dynamic imports by node builtins
- Updated dependencies [ab5adbe]
- Updated dependencies [195c428]
- Updated dependencies [f73e11b]
- Updated dependencies [422ee9e]
  - @mastra/core@0.10.2-alpha.3
  - @mastra/server@0.10.2-alpha.3

## 0.10.2-alpha.2

### Patch Changes

- e8d2aff: Fix non-scoped packages in mastra build
- 4187ed4: Fix mcp server api openapijson
- f0d559f: Fix peerdeps for alpha channel
- Updated dependencies [1e8bb40]
- Updated dependencies [f0d559f]
- Updated dependencies [a0ebc3f]
  - @mastra/core@0.10.2-alpha.2
  - @mastra/server@0.10.2-alpha.2

## 0.10.2-alpha.1

### Patch Changes

- Updated dependencies [ee77e78]
- Updated dependencies [2901125]
  - @mastra/core@0.10.2-alpha.1
  - @mastra/server@0.10.2-alpha.1

## 0.10.2-alpha.0

### Patch Changes

- Updated dependencies [592a2db]
- Updated dependencies [e5dc18d]
- Updated dependencies [e2228f6]
  - @mastra/core@0.10.2-alpha.0
  - @mastra/server@0.10.2-alpha.0

## 0.10.1

### Patch Changes

- 6d16390: Support custom bundle externals on mastra Instance
- bed0916: Handle wildcards in tools discovery
- 5343f93: Move emitter to symbol to make private
- fe68410: Fix mcp server routes
- Updated dependencies [d70b807]
- Updated dependencies [6d16390]
- Updated dependencies [1e4a421]
- Updated dependencies [200d0da]
- Updated dependencies [bf5f17b]
- Updated dependencies [5343f93]
- Updated dependencies [38aee50]
- Updated dependencies [5c41100]
- Updated dependencies [d6a759b]
- Updated dependencies [6015bdf]
  - @mastra/core@0.10.1
  - @mastra/server@0.10.1

## 0.10.1-alpha.3

### Patch Changes

- Updated dependencies [d70b807]
  - @mastra/core@0.10.1-alpha.3
  - @mastra/server@0.10.1-alpha.3

## 0.10.1-alpha.2

### Patch Changes

- fe68410: Fix mcp server routes
- Updated dependencies [6015bdf]
  - @mastra/server@0.10.1-alpha.1
  - @mastra/core@0.10.1-alpha.2

## 0.10.1-alpha.1

### Patch Changes

- bed0916: Handle wildcards in tools discovery
- 5343f93: Move emitter to symbol to make private
- Updated dependencies [200d0da]
- Updated dependencies [bf5f17b]
- Updated dependencies [5343f93]
- Updated dependencies [38aee50]
- Updated dependencies [5c41100]
- Updated dependencies [d6a759b]
  - @mastra/core@0.10.1-alpha.1
  - @mastra/server@0.10.1-alpha.0

## 0.10.1-alpha.0

### Patch Changes

- 6d16390: Support custom bundle externals on mastra Instance
- Updated dependencies [6d16390]
- Updated dependencies [1e4a421]
  - @mastra/core@0.10.1-alpha.0

## 0.10.0

### Minor Changes

- 83da932: Move @mastra/core to peerdeps
- 5eb5a99: Remove pino from @mastra/core into @mastra/loggers
- b2ae5aa: Added support for experimental authentication and authorization

### Patch Changes

- b3a3d63: BREAKING: Make vnext workflow the default worklow, and old workflow legacy_workflow
- 1e9fbfa: Upgrade to OpenTelemetry JS SDK 2.x
- 8d9feae: Add missing x-mastra-dev-playground headers
- aaf0e48: Add nodemailer to mastra bundler external deps
- 48e5910: Mastra server hostname, fallback to undefined
- 23f258c: Add new list and get routes for mcp servers. Changed route make-up for more consistency with existing API routes. Lastly, added in a lot of extra detail that can be optionally passed to the mcp server per the mcp spec.
- 2672a05: Add MCP servers and tool call execution to playground
- Updated dependencies [b3a3d63]
- Updated dependencies [344f453]
- Updated dependencies [0215b0b]
- Updated dependencies [0a3ae6d]
- Updated dependencies [95911be]
- Updated dependencies [83da932]
- Updated dependencies [f53a6ac]
- Updated dependencies [5eb5a99]
- Updated dependencies [7e632c5]
- Updated dependencies [1e9fbfa]
- Updated dependencies [eabdcd9]
- Updated dependencies [90be034]
- Updated dependencies [99f050a]
- Updated dependencies [d0ee3c6]
- Updated dependencies [b2ae5aa]
- Updated dependencies [23f258c]
- Updated dependencies [a7292b0]
- Updated dependencies [0dcb9f0]
- Updated dependencies [2672a05]
  - @mastra/server@0.10.0
  - @mastra/core@0.10.0

## 0.4.0-alpha.1

### Minor Changes

- 83da932: Move @mastra/core to peerdeps
- 5eb5a99: Remove pino from @mastra/core into @mastra/loggers
- b2ae5aa: Added support for experimental authentication and authorization

### Patch Changes

- b3a3d63: BREAKING: Make vnext workflow the default worklow, and old workflow legacy_workflow
- 1e9fbfa: Upgrade to OpenTelemetry JS SDK 2.x
- 8d9feae: Add missing x-mastra-dev-playground headers
- Updated dependencies [b3a3d63]
- Updated dependencies [344f453]
- Updated dependencies [0215b0b]
- Updated dependencies [0a3ae6d]
- Updated dependencies [95911be]
- Updated dependencies [83da932]
- Updated dependencies [5eb5a99]
- Updated dependencies [7e632c5]
- Updated dependencies [1e9fbfa]
- Updated dependencies [b2ae5aa]
- Updated dependencies [a7292b0]
- Updated dependencies [0dcb9f0]
  - @mastra/server@2.1.0-alpha.1
  - @mastra/core@0.10.0-alpha.1

## 0.3.5-alpha.0

### Patch Changes

- aaf0e48: Add nodemailer to mastra bundler external deps
- 48e5910: Mastra server hostname, fallback to undefined
- 23f258c: Add new list and get routes for mcp servers. Changed route make-up for more consistency with existing API routes. Lastly, added in a lot of extra detail that can be optionally passed to the mcp server per the mcp spec.
- 2672a05: Add MCP servers and tool call execution to playground
- Updated dependencies [f53a6ac]
- Updated dependencies [eabdcd9]
- Updated dependencies [90be034]
- Updated dependencies [99f050a]
- Updated dependencies [d0ee3c6]
- Updated dependencies [23f258c]
- Updated dependencies [2672a05]
  - @mastra/server@2.0.5-alpha.0
  - @mastra/core@0.9.5-alpha.0

## 0.3.4

### Patch Changes

- 396be50: updated mcp server routes for MCP SSE for use with hono server
- 5c70b8a: [MASTRA-3234] added limit for client-js getMessages
- 03c40d1: instructions is only available in playground
- cb1f698: Set runtimeContext from playground for agents, tools, workflows
- 0b8b868: Added A2A support + streaming
- edf1e88: allows ability to pass McpServer into the mastra class and creates an endpoint /api/servers/:serverId/mcp to POST messages to an MCP server
- Updated dependencies [396be50]
- Updated dependencies [ab80e7e]
- Updated dependencies [5c70b8a]
- Updated dependencies [c3bd795]
- Updated dependencies [da082f8]
- Updated dependencies [0c3d117]
- Updated dependencies [a5810ce]
- Updated dependencies [3e9c131]
- Updated dependencies [3171b5b]
- Updated dependencies [cb1f698]
- Updated dependencies [973e5ac]
- Updated dependencies [daf942f]
- Updated dependencies [0b8b868]
- Updated dependencies [9e1eff5]
- Updated dependencies [6fa1ad1]
- Updated dependencies [c28d7a0]
- Updated dependencies [edf1e88]
  - @mastra/core@0.9.4
  - @mastra/server@2.0.4

## 0.3.4-alpha.4

### Patch Changes

- 5c70b8a: [MASTRA-3234] added limit for client-js getMessages
- Updated dependencies [5c70b8a]
- Updated dependencies [3e9c131]
  - @mastra/server@2.0.4-alpha.4
  - @mastra/core@0.9.4-alpha.4

## 0.3.4-alpha.3

### Patch Changes

- 396be50: updated mcp server routes for MCP SSE for use with hono server
- Updated dependencies [396be50]
- Updated dependencies [c3bd795]
- Updated dependencies [da082f8]
- Updated dependencies [0c3d117]
- Updated dependencies [a5810ce]
  - @mastra/core@0.9.4-alpha.3
  - @mastra/server@2.0.4-alpha.3

## 0.3.4-alpha.2

### Patch Changes

- 03c40d1: instructions is only available in playground
- Updated dependencies [3171b5b]
- Updated dependencies [973e5ac]
- Updated dependencies [9e1eff5]
  - @mastra/core@0.9.4-alpha.2
  - @mastra/server@2.0.4-alpha.2

## 0.3.4-alpha.1

### Patch Changes

- edf1e88: allows ability to pass McpServer into the mastra class and creates an endpoint /api/servers/:serverId/mcp to POST messages to an MCP server
- Updated dependencies [ab80e7e]
- Updated dependencies [6fa1ad1]
- Updated dependencies [c28d7a0]
- Updated dependencies [edf1e88]
  - @mastra/server@2.0.4-alpha.1
  - @mastra/core@0.9.4-alpha.1

## 0.3.4-alpha.0

### Patch Changes

- cb1f698: Set runtimeContext from playground for agents, tools, workflows
- 0b8b868: Added A2A support + streaming
- Updated dependencies [cb1f698]
- Updated dependencies [daf942f]
- Updated dependencies [0b8b868]
  - @mastra/server@2.0.4-alpha.0
  - @mastra/core@0.9.4-alpha.0

## 0.3.3

### Patch Changes

- 8902157: added an optional `bodySizeLimit` to server config so that users can pass custom bodylimit size in mb. If not, it defaults to 4.5 mb
- 70dbf51: [MASTRA-2452] updated setBaggage for tracing
- Updated dependencies [e450778]
- Updated dependencies [8902157]
- Updated dependencies [ca0dc88]
- Updated dependencies [526c570]
- Updated dependencies [d7a6a33]
- Updated dependencies [9cd1a46]
- Updated dependencies [b5d2de0]
- Updated dependencies [644f8ad]
- Updated dependencies [70dbf51]
  - @mastra/core@0.9.3
  - @mastra/server@2.0.3

## 0.3.3-alpha.1

### Patch Changes

- 8902157: added an optional `bodySizeLimit` to server config so that users can pass custom bodylimit size in mb. If not, it defaults to 4.5 mb
- 70dbf51: [MASTRA-2452] updated setBaggage for tracing
- Updated dependencies [e450778]
- Updated dependencies [8902157]
- Updated dependencies [ca0dc88]
- Updated dependencies [9cd1a46]
- Updated dependencies [70dbf51]
  - @mastra/core@0.9.3-alpha.1
  - @mastra/server@2.0.3-alpha.1

## 0.3.3-alpha.0

### Patch Changes

- Updated dependencies [526c570]
- Updated dependencies [b5d2de0]
- Updated dependencies [644f8ad]
  - @mastra/server@2.0.3-alpha.0
  - @mastra/core@0.9.3-alpha.0

## 0.3.2

### Patch Changes

- 2cf3b8f: dependencies updates:
  - Updated dependency [`zod@^3.24.3` ↗︎](https://www.npmjs.com/package/zod/v/3.24.3) (from `^3.24.2`, in `dependencies`)
- 4155f47: Add parameters to filter workflow runs
  Add fromDate and toDate to telemetry parameters
- 254f5c3: Audit, cleanup MastraClient
- 8607972: Introduce Mastra lint cli command
- a798090: Do not break on tools not being to import
- Updated dependencies [6052aa6]
- Updated dependencies [967b41c]
- Updated dependencies [3d2fb5c]
- Updated dependencies [26738f4]
- Updated dependencies [4155f47]
- Updated dependencies [7eeb2bc]
- Updated dependencies [b804723]
- Updated dependencies [8607972]
- Updated dependencies [ccef9f9]
- Updated dependencies [0097d50]
- Updated dependencies [7eeb2bc]
- Updated dependencies [17826a9]
- Updated dependencies [7d8b7c7]
- Updated dependencies [fba031f]
- Updated dependencies [3a5f1e1]
- Updated dependencies [51e6923]
- Updated dependencies [8398d89]
  - @mastra/server@2.0.2
  - @mastra/core@0.9.2

## 0.3.2-alpha.6

### Patch Changes

- a798090: Do not break on tools not being to import
- Updated dependencies [6052aa6]
- Updated dependencies [7d8b7c7]
- Updated dependencies [3a5f1e1]
- Updated dependencies [8398d89]
  - @mastra/server@2.0.2-alpha.6
  - @mastra/core@0.9.2-alpha.6

## 0.3.2-alpha.5

### Patch Changes

- 8607972: Introduce Mastra lint cli command
- Updated dependencies [3d2fb5c]
- Updated dependencies [7eeb2bc]
- Updated dependencies [8607972]
- Updated dependencies [7eeb2bc]
- Updated dependencies [fba031f]
  - @mastra/core@0.9.2-alpha.5
  - @mastra/server@2.0.2-alpha.5

## 0.3.2-alpha.4

### Patch Changes

- Updated dependencies [ccef9f9]
- Updated dependencies [51e6923]
  - @mastra/core@0.9.2-alpha.4
  - @mastra/server@2.0.2-alpha.4

## 0.3.2-alpha.3

### Patch Changes

- 4155f47: Add parameters to filter workflow runs
  Add fromDate and toDate to telemetry parameters
- Updated dependencies [967b41c]
- Updated dependencies [4155f47]
- Updated dependencies [17826a9]
  - @mastra/core@0.9.2-alpha.3
  - @mastra/server@2.0.2-alpha.3

## 0.3.2-alpha.2

### Patch Changes

- Updated dependencies [26738f4]
  - @mastra/core@0.9.2-alpha.2
  - @mastra/server@2.0.2-alpha.2

## 0.3.2-alpha.1

### Patch Changes

- 254f5c3: Audit, cleanup MastraClient
- Updated dependencies [b804723]
  - @mastra/core@0.9.2-alpha.1
  - @mastra/server@2.0.2-alpha.1

## 0.3.2-alpha.0

### Patch Changes

- Updated dependencies [0097d50]
  - @mastra/server@2.0.2-alpha.0
  - @mastra/core@0.9.2-alpha.0

## 0.3.1

### Patch Changes

- e7c2881: fix: support dynamic imports when bundling
- 0ccb8b4: Fix deployer bundling when custom mastra dir is set
- 92c598d: Remove API request logs from local dev server
- ebdb781: Fix writing tools in correct folder
- 35955b0: Rename import to runtime-contxt
- 6262bd5: Mastra server custom host config
- c1409ef: Add vNextWorkflow handlers and APIs
  Add stepGraph and steps to vNextWorkflow
- 3e7b69d: Dynamic agent props
- 11d4485: Show VNext workflows on the playground
  Show running status for step in vNext workflowState
- 530ced1: Fix cloudflare deployer by removing import.meta.url reference
- 611aa4a: add all builds to run postinstall
- 1d3b1cd: Rebump
- Updated dependencies [34a76ca]
- Updated dependencies [405b63d]
- Updated dependencies [81fb7f6]
- Updated dependencies [20275d4]
- Updated dependencies [7d1892c]
- Updated dependencies [a90a082]
- Updated dependencies [2d17c73]
- Updated dependencies [61e92f5]
- Updated dependencies [35955b0]
- Updated dependencies [6262bd5]
- Updated dependencies [c1409ef]
- Updated dependencies [3e7b69d]
- Updated dependencies [e4943b8]
- Updated dependencies [f200fed]
- Updated dependencies [11d4485]
- Updated dependencies [479f490]
- Updated dependencies [57b25ed]
- Updated dependencies [c23a81c]
- Updated dependencies [2d4001d]
- Updated dependencies [c71013a]
- Updated dependencies [1d3b1cd]
  - @mastra/server@2.0.1
  - @mastra/core@0.9.1

## 0.3.1-alpha.8

### Patch Changes

- Updated dependencies [2d17c73]
  - @mastra/core@0.9.1-alpha.8
  - @mastra/server@2.0.1-alpha.8

## 0.3.1-alpha.7

### Patch Changes

- 1d3b1cd: Rebump
- Updated dependencies [1d3b1cd]
  - @mastra/core@0.9.1-alpha.7
  - @mastra/server@2.0.1-alpha.7

## 0.3.1-alpha.6

### Patch Changes

- Updated dependencies [c23a81c]
  - @mastra/core@0.9.1-alpha.6
  - @mastra/server@2.0.1-alpha.6

## 0.3.1-alpha.5

### Patch Changes

- 3e7b69d: Dynamic agent props
- Updated dependencies [3e7b69d]
  - @mastra/core@0.9.1-alpha.5
  - @mastra/server@2.0.1-alpha.5

## 0.3.1-alpha.4

### Patch Changes

- Updated dependencies [e4943b8]
- Updated dependencies [479f490]
  - @mastra/core@0.9.1-alpha.4
  - @mastra/server@2.0.1-alpha.4

## 0.3.1-alpha.3

### Patch Changes

- 6262bd5: Mastra server custom host config
- Updated dependencies [34a76ca]
- Updated dependencies [6262bd5]
  - @mastra/server@2.0.1-alpha.3
  - @mastra/core@0.9.1-alpha.3

## 0.3.1-alpha.2

### Patch Changes

- Updated dependencies [405b63d]
- Updated dependencies [61e92f5]
- Updated dependencies [57b25ed]
- Updated dependencies [c71013a]
  - @mastra/core@0.9.1-alpha.2
  - @mastra/server@2.0.1-alpha.2

## 0.3.1-alpha.1

### Patch Changes

- e7c2881: fix: support dynamic imports when bundling
- 0ccb8b4: Fix deployer bundling when custom mastra dir is set
- 92c598d: Remove API request logs from local dev server
- ebdb781: Fix writing tools in correct folder
- 35955b0: Rename import to runtime-contxt
- c1409ef: Add vNextWorkflow handlers and APIs
  Add stepGraph and steps to vNextWorkflow
- 11d4485: Show VNext workflows on the playground
  Show running status for step in vNext workflowState
- 530ced1: Fix cloudflare deployer by removing import.meta.url reference
- 611aa4a: add all builds to run postinstall
- Updated dependencies [20275d4]
- Updated dependencies [7d1892c]
- Updated dependencies [a90a082]
- Updated dependencies [35955b0]
- Updated dependencies [c1409ef]
- Updated dependencies [f200fed]
- Updated dependencies [11d4485]
- Updated dependencies [2d4001d]
  - @mastra/core@0.9.1-alpha.1
  - @mastra/server@2.0.1-alpha.1

## 0.3.1-alpha.0

### Patch Changes

- Updated dependencies [81fb7f6]
  - @mastra/core@0.9.1-alpha.0
  - @mastra/server@2.0.1-alpha.0

## 0.3.0

### Minor Changes

- fe3ae4d: Remove \_\_ functions in storage and move to storage proxy to make sure init is called

### Patch Changes

- b9122b0: fix: When using a third party exporter such as Langfuse we were not installing external deps imported from the telemetry config
- 3527610: Fix multi slash imports during bundling
- 7e92011: Include tools with deployment builds
- 2538066: Fix memory thread creation from client SDK
- 63fe16a: Support monorepo workspace packages with native bindings
- 0f4eae3: Rename Container into RuntimeContext
- 3f9d151: Add support for tsconfig paths in server-configuration
- 735ead7: Add support for process.env.development
- 16a8648: Disable swaggerUI, playground for production builds, mastra instance server build config to enable swaggerUI, apiReqLogs, openAPI documentation for prod builds
- Updated dependencies [000a6d4]
- Updated dependencies [08bb78e]
- Updated dependencies [ed2f549]
- Updated dependencies [7e92011]
- Updated dependencies [9ee4293]
- Updated dependencies [03f3cd0]
- Updated dependencies [c0f22b4]
- Updated dependencies [71d9444]
- Updated dependencies [157c741]
- Updated dependencies [8a8a73b]
- Updated dependencies [0a033fa]
- Updated dependencies [fe3ae4d]
- Updated dependencies [9c26508]
- Updated dependencies [0f4eae3]
- Updated dependencies [1c0d2b7]
- Updated dependencies [16a8648]
- Updated dependencies [6f92295]
  - @mastra/core@0.9.0
  - @mastra/server@2.0.0

## 0.3.0-alpha.9

### Patch Changes

- b9122b0: fix: When using a third party exporter such as Langfuse we were not installing external deps imported from the telemetry config
- 2538066: Fix memory thread creation from client SDK
- 0f4eae3: Rename Container into RuntimeContext
- 16a8648: Disable swaggerUI, playground for production builds, mastra instance server build config to enable swaggerUI, apiReqLogs, openAPI documentation for prod builds
- Updated dependencies [000a6d4]
- Updated dependencies [ed2f549]
- Updated dependencies [c0f22b4]
- Updated dependencies [0a033fa]
- Updated dependencies [9c26508]
- Updated dependencies [0f4eae3]
- Updated dependencies [1c0d2b7]
- Updated dependencies [16a8648]
  - @mastra/core@0.9.0-alpha.8
  - @mastra/server@2.0.0-alpha.8

## 0.3.0-alpha.8

### Patch Changes

- Updated dependencies [71d9444]
  - @mastra/core@0.9.0-alpha.7
  - @mastra/server@2.0.0-alpha.7

## 0.3.0-alpha.7

### Patch Changes

- 63fe16a: Support monorepo workspace packages with native bindings
- 735ead7: Add support for process.env.development
- Updated dependencies [157c741]
  - @mastra/core@0.9.0-alpha.6
  - @mastra/server@2.0.0-alpha.6

## 0.3.0-alpha.6

### Patch Changes

- 3f9d151: Add support for tsconfig paths in server-configuration
- Updated dependencies [08bb78e]
  - @mastra/core@0.9.0-alpha.5
  - @mastra/server@2.0.0-alpha.5

## 0.3.0-alpha.5

### Patch Changes

- 7e92011: Include tools with deployment builds
- Updated dependencies [7e92011]
  - @mastra/core@0.9.0-alpha.4
  - @mastra/server@2.0.0-alpha.4

## 0.3.0-alpha.4

### Minor Changes

- fe3ae4d: Remove \_\_ functions in storage and move to storage proxy to make sure init is called

### Patch Changes

- Updated dependencies [fe3ae4d]
  - @mastra/server@2.0.0-alpha.3
  - @mastra/core@0.9.0-alpha.3

## 0.2.10-alpha.3

### Patch Changes

- Updated dependencies [9ee4293]
  - @mastra/core@0.8.4-alpha.2
  - @mastra/server@1.0.4-alpha.2

## 0.2.10-alpha.2

### Patch Changes

- 3527610: Fix multi slash imports during bundling

## 0.2.10-alpha.1

### Patch Changes

- Updated dependencies [8a8a73b]
- Updated dependencies [6f92295]
  - @mastra/core@0.8.4-alpha.1
  - @mastra/server@1.0.4-alpha.1

## 0.2.10-alpha.0

### Patch Changes

- Updated dependencies [03f3cd0]
  - @mastra/core@0.8.4-alpha.0
  - @mastra/server@1.0.4-alpha.0

## 0.2.9

### Patch Changes

- 9f6f6dd: Fix container for tools execution api
- 32e7b71: Add support for dependency injection
- 37bb612: Add Elastic-2.0 licensing for packages
- 1ebbfbf: Add 3 minutes timeout to deployer server
- 67aff42: Fix netlify deployer missing @libsql/linux-x64-gnu bug
- Updated dependencies [d72318f]
- Updated dependencies [0bcc862]
- Updated dependencies [10a8caf]
- Updated dependencies [359b089]
- Updated dependencies [9f6f6dd]
- Updated dependencies [32e7b71]
- Updated dependencies [37bb612]
- Updated dependencies [7f1b291]
  - @mastra/core@0.8.3
  - @mastra/server@1.0.3

## 0.2.9-alpha.7

### Patch Changes

- Updated dependencies [d72318f]
  - @mastra/core@0.8.3-alpha.5
  - @mastra/server@1.0.3-alpha.6

## 0.2.9-alpha.6

### Patch Changes

- 67aff42: Fix netlify deployer missing @libsql/linux-x64-gnu bug

## 0.2.9-alpha.5

### Patch Changes

- 9f6f6dd: Fix container for tools execution api
- Updated dependencies [9f6f6dd]
  - @mastra/server@1.0.3-alpha.5

## 0.2.9-alpha.4

### Patch Changes

- 1ebbfbf: Add 3 minutes timeout to deployer server
- Updated dependencies [7f1b291]
  - @mastra/core@0.8.3-alpha.4
  - @mastra/server@1.0.3-alpha.4

## 0.2.9-alpha.3

### Patch Changes

- Updated dependencies [10a8caf]
  - @mastra/core@0.8.3-alpha.3
  - @mastra/server@1.0.3-alpha.3

## 0.2.9-alpha.2

### Patch Changes

- Updated dependencies [0bcc862]
  - @mastra/core@0.8.3-alpha.2
  - @mastra/server@1.0.3-alpha.2

## 0.2.9-alpha.1

### Patch Changes

- 32e7b71: Add support for dependency injection
- 37bb612: Add Elastic-2.0 licensing for packages
- Updated dependencies [32e7b71]
- Updated dependencies [37bb612]
  - @mastra/server@1.0.3-alpha.1
  - @mastra/core@0.8.3-alpha.1

## 0.2.9-alpha.0

### Patch Changes

- Updated dependencies [359b089]
  - @mastra/core@0.8.3-alpha.0
  - @mastra/server@1.0.3-alpha.0

## 0.2.8

### Patch Changes

- ae6c5ce: Fix await loop inside mastra entrypoint
- 94cd5c1: Fix yarn workspace isolation
- Updated dependencies [a06aadc]
  - @mastra/core@0.8.2
  - @mastra/server@1.0.2

## 0.2.8-alpha.1

### Patch Changes

- 94cd5c1: Fix yarn workspace isolation

## 0.2.8-alpha.0

### Patch Changes

- ae6c5ce: Fix await loop inside mastra entrypoint
- Updated dependencies [a06aadc]
  - @mastra/core@0.8.2-alpha.0
  - @mastra/server@1.0.2-alpha.0

## 0.2.7

### Patch Changes

- 8fdb414: Custom mastra server cors config
- Updated dependencies [99e2998]
- Updated dependencies [8fdb414]
  - @mastra/core@0.8.1
  - @mastra/server@1.0.1

## 0.2.7-alpha.0

### Patch Changes

- 8fdb414: Custom mastra server cors config
- Updated dependencies [99e2998]
- Updated dependencies [8fdb414]
  - @mastra/core@0.8.1-alpha.0
  - @mastra/server@1.0.1-alpha.0

## 0.2.6

### Patch Changes

- 2135c81: Alias @mastra/server in bundler
- 05d58cc: fix: add 'x-mastra-client-type' to allowed headers in CORS configuration
- 4c98129: Upgrade babel-core
- 4c65a57: Add fastebmed as external
- 84fe241: Decoupled handlers from hono
- 88fa727: Added getWorkflowRuns for libsql, pg, clickhouse and upstash as well as added route getWorkflowRunsHandler
- dfb0601: Add missing triggerData to the openapi.json for the POST /api/workflow/{workflowId}/start endpoint
- 789bef3: Make runId optional for workflow startAsync api
- a3f0e90: Update storage initialization to ensure tables are present
- 6330967: Enable route timeout using server options
- 8393832: Handle nested workflow view on workflow graph
- 6330967: Add support for configuration of server port using Mastra instance
- 84fe241: Improve streaming of workflows
- 32ba03c: Make timeout 30s
- 3c6ae54: Fix fastembed part of dependencies
- febc8a6: Added dual tracing and fixed local tracing recursion
- 0deb356: Fixed a bug where the hono body wasn't properly passed into stream+generate API handlers resulting in "cannot destructure property messages of body"
- 8076ecf: Unify workflow watch/start response
- 304397c: Add support for custom api routes in mastra
- Updated dependencies [56c31b7]
- Updated dependencies [619c39d]
- Updated dependencies [5ae0180]
- Updated dependencies [fe56be0]
- Updated dependencies [93875ed]
- Updated dependencies [107bcfe]
- Updated dependencies [9bfa12b]
- Updated dependencies [515ebfb]
- Updated dependencies [5b4e19f]
- Updated dependencies [dbbbf80]
- Updated dependencies [a0967a0]
- Updated dependencies [84fe241]
- Updated dependencies [fca3b21]
- Updated dependencies [88fa727]
- Updated dependencies [f37f535]
- Updated dependencies [789bef3]
- Updated dependencies [a3f0e90]
- Updated dependencies [4d67826]
- Updated dependencies [6330967]
- Updated dependencies [8393832]
- Updated dependencies [6330967]
- Updated dependencies [84fe241]
- Updated dependencies [99d43b9]
- Updated dependencies [d7e08e8]
- Updated dependencies [febc8a6]
- Updated dependencies [7599d77]
- Updated dependencies [0118361]
- Updated dependencies [619c39d]
- Updated dependencies [cafae83]
- Updated dependencies [8076ecf]
- Updated dependencies [8df4a77]
- Updated dependencies [304397c]
  - @mastra/core@0.8.0
  - @mastra/server@1.0.0

## 0.2.6-alpha.10

### Patch Changes

- 2135c81: Alias @mastra/server in bundler
- Updated dependencies [8df4a77]
  - @mastra/core@0.8.0-alpha.8
  - @mastra/server@0.0.1-alpha.6

## 0.2.6-alpha.9

### Patch Changes

- 3c6ae54: Fix fastembed part of dependencies
- febc8a6: Added dual tracing and fixed local tracing recursion
- Updated dependencies [febc8a6]
  - @mastra/server@0.0.1-alpha.5
  - @mastra/core@0.8.0-alpha.7

## 0.2.6-alpha.8

### Patch Changes

- 4c65a57: Add fastebmed as external
- a3f0e90: Update storage initialization to ensure tables are present
- Updated dependencies [a3f0e90]
  - @mastra/server@0.0.1-alpha.4
  - @mastra/core@0.8.0-alpha.6

## 0.2.6-alpha.7

### Patch Changes

- Updated dependencies [93875ed]
  - @mastra/core@0.8.0-alpha.5
  - @mastra/server@0.0.1-alpha.3

## 0.2.6-alpha.6

### Patch Changes

- Updated dependencies [d7e08e8]
  - @mastra/core@0.8.0-alpha.4
  - @mastra/server@0.0.1-alpha.2

## 0.2.6-alpha.5

### Patch Changes

- 32ba03c: Make timeout 30s

## 0.2.6-alpha.4

### Patch Changes

- 88fa727: Added getWorkflowRuns for libsql, pg, clickhouse and upstash as well as added route getWorkflowRunsHandler
- dfb0601: Add missing triggerData to the openapi.json for the POST /api/workflow/{workflowId}/start endpoint
- 789bef3: Make runId optional for workflow startAsync api
- 6330967: Enable route timeout using server options
- 8393832: Handle nested workflow view on workflow graph
- 6330967: Add support for configuration of server port using Mastra instance
- Updated dependencies [5ae0180]
- Updated dependencies [9bfa12b]
- Updated dependencies [515ebfb]
- Updated dependencies [88fa727]
- Updated dependencies [f37f535]
- Updated dependencies [789bef3]
- Updated dependencies [4d67826]
- Updated dependencies [6330967]
- Updated dependencies [8393832]
- Updated dependencies [6330967]
  - @mastra/core@0.8.0-alpha.3
  - @mastra/server@0.0.1-alpha.1

## 0.2.6-alpha.3

### Patch Changes

- 0deb356: Fixed a bug where the hono body wasn't properly passed into stream+generate API handlers resulting in "cannot destructure property messages of body"

## 0.2.6-alpha.2

### Patch Changes

- 4c98129: Upgrade babel-core
- 84fe241: Decoupled handlers from hono
- 84fe241: Improve streaming of workflows
- Updated dependencies [56c31b7]
- Updated dependencies [dbbbf80]
- Updated dependencies [84fe241]
- Updated dependencies [84fe241]
- Updated dependencies [99d43b9]
  - @mastra/core@0.8.0-alpha.2
  - @mastra/server@0.0.1-alpha.0

## 0.2.6-alpha.1

### Patch Changes

- Updated dependencies [619c39d]
- Updated dependencies [fe56be0]
- Updated dependencies [a0967a0]
- Updated dependencies [fca3b21]
- Updated dependencies [0118361]
- Updated dependencies [619c39d]
  - @mastra/core@0.8.0-alpha.1

## 0.2.6-alpha.0

### Patch Changes

- 05d58cc: fix: add 'x-mastra-client-type' to allowed headers in CORS configuration
- 8076ecf: Unify workflow watch/start response
- 304397c: Add support for custom api routes in mastra
- Updated dependencies [107bcfe]
- Updated dependencies [5b4e19f]
- Updated dependencies [7599d77]
- Updated dependencies [cafae83]
- Updated dependencies [8076ecf]
- Updated dependencies [304397c]
  - @mastra/core@0.7.1-alpha.0

## 0.2.5

### Patch Changes

- cdc0498: Fix process.versions.node.split in cloudflare deployer
- 0b496ff: Load env vars on mastra deploy
- Updated dependencies [b4fbc59]
- Updated dependencies [a838fde]
- Updated dependencies [a8bd4cf]
- Updated dependencies [7a3eeb0]
- Updated dependencies [0b54522]
- Updated dependencies [b3b34f5]
- Updated dependencies [1af25d5]
- Updated dependencies [a4686e8]
- Updated dependencies [6530ad1]
- Updated dependencies [27439ad]
  - @mastra/core@0.7.0

## 0.2.5-alpha.3

### Patch Changes

- Updated dependencies [b3b34f5]
- Updated dependencies [a4686e8]
  - @mastra/core@0.7.0-alpha.3

## 0.2.5-alpha.2

### Patch Changes

- Updated dependencies [a838fde]
- Updated dependencies [a8bd4cf]
- Updated dependencies [7a3eeb0]
- Updated dependencies [6530ad1]
  - @mastra/core@0.7.0-alpha.2

## 0.2.5-alpha.1

### Patch Changes

- cdc0498: Fix process.versions.node.split in cloudflare deployer
- 0b496ff: Load env vars on mastra deploy
- Updated dependencies [0b54522]
- Updated dependencies [1af25d5]
- Updated dependencies [27439ad]
  - @mastra/core@0.7.0-alpha.1

## 0.2.5-alpha.0

### Patch Changes

- Updated dependencies [b4fbc59]
  - @mastra/core@0.6.5-alpha.0

## 0.2.4

### Patch Changes

- e764fd1: Fix telemetry when side-effects are added to the mastra file
- 709aa2c: fix building externals
- e764fd1: Fix deployer when side-effects are added to the mastra file
- 05ef3e0: Support voice for mastra client
- 95c5745: Fix symlink resolving and externals
- 85a2461: Fix cloudflare deployer
- Updated dependencies [6794797]
- Updated dependencies [fb68a80]
- Updated dependencies [b56a681]
- Updated dependencies [248cb07]
  - @mastra/core@0.6.4

## 0.2.4-alpha.1

### Patch Changes

- 709aa2c: fix building externals
- 85a2461: Fix cloudflare deployer
- Updated dependencies [6794797]
  - @mastra/core@0.6.4-alpha.1

## 0.2.4-alpha.0

### Patch Changes

- e764fd1: Fix telemetry when side-effects are added to the mastra file
- e764fd1: Fix deployer when side-effects are added to the mastra file
- 05ef3e0: Support voice for mastra client
- 95c5745: Fix symlink resolving and externals
- Updated dependencies [fb68a80]
- Updated dependencies [b56a681]
- Updated dependencies [248cb07]
  - @mastra/core@0.6.4-alpha.0

## 0.2.3

### Patch Changes

- 404640e: AgentNetwork changeset
- Updated dependencies [404640e]
- Updated dependencies [3bce733]
  - @mastra/core@0.6.3

## 0.2.3-alpha.1

### Patch Changes

- Updated dependencies [3bce733]
  - @mastra/core@0.6.3-alpha.1

## 0.2.3-alpha.0

### Patch Changes

- 404640e: AgentNetwork changeset
- Updated dependencies [404640e]
  - @mastra/core@0.6.3-alpha.0

## 0.2.2

### Patch Changes

- 4e6732b: Add support for tsconfig paths aliases
- Updated dependencies [beaf1c2]
- Updated dependencies [3084e13]
  - @mastra/core@0.6.2

## 0.2.2-alpha.1

### Patch Changes

- Updated dependencies [beaf1c2]
- Updated dependencies [3084e13]
  - @mastra/core@0.6.2-alpha.0

## 0.2.2-alpha.0

### Patch Changes

- 4e6732b: Add support for tsconfig paths aliases

## 0.2.1

### Patch Changes

- cc7f392: Fix babel transformation in deployer
- 0850b4c: Watch and resume per run
- da8d9bb: Enable public dir copying if it exists
- 9116d70: Handle the different workflow methods in workflow graph
- 61ad5a4: Move esbuild plugin higher than commonjs for telemetry extraction
- Updated dependencies [fc2f89c]
- Updated dependencies [dfbb131]
- Updated dependencies [f4854ee]
- Updated dependencies [afaf73f]
- Updated dependencies [0850b4c]
- Updated dependencies [7bcfaee]
- Updated dependencies [44631b1]
- Updated dependencies [9116d70]
- Updated dependencies [6e559a0]
- Updated dependencies [5f43505]
  - @mastra/core@0.6.1

## 0.2.1-alpha.2

### Patch Changes

- cc7f392: Fix babel transformation in deployer
- 0850b4c: Watch and resume per run
- da8d9bb: Enable public dir copying if it exists
- 9116d70: Handle the different workflow methods in workflow graph
- Updated dependencies [fc2f89c]
- Updated dependencies [dfbb131]
- Updated dependencies [0850b4c]
- Updated dependencies [9116d70]
  - @mastra/core@0.6.1-alpha.2

## 0.2.1-alpha.1

### Patch Changes

- 61ad5a4: Move esbuild plugin higher than commonjs for telemetry extraction
- Updated dependencies [f4854ee]
- Updated dependencies [afaf73f]
- Updated dependencies [44631b1]
- Updated dependencies [6e559a0]
- Updated dependencies [5f43505]
  - @mastra/core@0.6.1-alpha.1

## 0.2.1-alpha.0

### Patch Changes

- Updated dependencies [7bcfaee]
  - @mastra/core@0.6.1-alpha.0

## 0.2.0

### Minor Changes

- 95b4144: Added server middleware to apply custom functionality in API endpoints like auth

### Patch Changes

- Updated dependencies [16b98d9]
- Updated dependencies [1c8cda4]
- Updated dependencies [95b4144]
- Updated dependencies [3729dbd]
- Updated dependencies [c2144f4]
  - @mastra/core@0.6.0

## 0.2.0-alpha.1

### Minor Changes

- 95b4144: Added server middleware to apply custom functionality in API endpoints like auth

### Patch Changes

- Updated dependencies [16b98d9]
- Updated dependencies [1c8cda4]
- Updated dependencies [95b4144]
- Updated dependencies [c2144f4]
  - @mastra/core@0.6.0-alpha.1

## 0.1.9-alpha.0

### Patch Changes

- Updated dependencies [3729dbd]
  - @mastra/core@0.5.1-alpha.0

## 0.1.8

### Patch Changes

- 7a7a547: Fix telemetry getter in hono server
- e9fbac5: Update Vercel tools to have id and update deployer
- 8deb34c: Better workflow watch api + watch workflow by runId
- c2dde91: Return full workflow details in api/workflows endpoint
- 5d41958: Remove redundant mastra server agent stream, generate messages validation
- 144b3d5: Update traces table UI, agent Chat UI
  Fix get workflows breaking
- 03236ec: Added GRPC Exporter for Laminar and updated dodcs for Observability Providers
- 731dd8a: Removed useless logging that showed up when user selected log drains tab on the playground
- 0461849: Fixed a bug where mastra.db file location was inconsistently created when running mastra dev vs running a file directly (tsx src/index.ts for ex)
- fd4a1d7: Update cjs bundling to make sure files are split
- 960690d: return runId from server on workflow watch
- Updated dependencies [a910463]
- Updated dependencies [59df7b6]
- Updated dependencies [22643eb]
- Updated dependencies [6feb23f]
- Updated dependencies [f2d6727]
- Updated dependencies [7a7a547]
- Updated dependencies [29f3a82]
- Updated dependencies [3d0e290]
- Updated dependencies [e9fbac5]
- Updated dependencies [301e4ee]
- Updated dependencies [ee667a2]
- Updated dependencies [dfbe4e9]
- Updated dependencies [dab255b]
- Updated dependencies [1e8bcbc]
- Updated dependencies [f6678e4]
- Updated dependencies [9e81f35]
- Updated dependencies [c93798b]
- Updated dependencies [a85ab24]
- Updated dependencies [dbd9f2d]
- Updated dependencies [59df7b6]
- Updated dependencies [caefaa2]
- Updated dependencies [c151ae6]
- Updated dependencies [52e0418]
- Updated dependencies [d79aedf]
- Updated dependencies [03236ec]
- Updated dependencies [3764e71]
- Updated dependencies [df982db]
- Updated dependencies [a171b37]
- Updated dependencies [506f1d5]
- Updated dependencies [02ffb7b]
- Updated dependencies [0461849]
- Updated dependencies [2259379]
- Updated dependencies [aeb5e36]
- Updated dependencies [f2301de]
- Updated dependencies [358f069]
- Updated dependencies [fd4a1d7]
- Updated dependencies [c139344]
  - @mastra/core@0.5.0

## 0.1.8-alpha.12

### Patch Changes

- Updated dependencies [a85ab24]
  - @mastra/core@0.5.0-alpha.12

## 0.1.8-alpha.11

### Patch Changes

- 7a7a547: Fix telemetry getter in hono server
- 8deb34c: Better workflow watch api + watch workflow by runId
- 5d41958: Remove redundant mastra server agent stream, generate messages validation
- fd4a1d7: Update cjs bundling to make sure files are split
- Updated dependencies [7a7a547]
- Updated dependencies [c93798b]
- Updated dependencies [dbd9f2d]
- Updated dependencies [a171b37]
- Updated dependencies [fd4a1d7]
  - @mastra/core@0.5.0-alpha.11

## 0.1.8-alpha.10

### Patch Changes

- Updated dependencies [a910463]
  - @mastra/core@0.5.0-alpha.10

## 0.1.8-alpha.9

### Patch Changes

- e9fbac5: Update Vercel tools to have id and update deployer
- Updated dependencies [e9fbac5]
- Updated dependencies [1e8bcbc]
- Updated dependencies [aeb5e36]
- Updated dependencies [f2301de]
  - @mastra/core@0.5.0-alpha.9

## 0.1.8-alpha.8

### Patch Changes

- Updated dependencies [506f1d5]
  - @mastra/core@0.5.0-alpha.8

## 0.1.8-alpha.7

### Patch Changes

- Updated dependencies [ee667a2]
  - @mastra/core@0.5.0-alpha.7

## 0.1.8-alpha.6

### Patch Changes

- Updated dependencies [f6678e4]
  - @mastra/core@0.5.0-alpha.6

## 0.1.8-alpha.5

### Patch Changes

- 03236ec: Added GRPC Exporter for Laminar and updated dodcs for Observability Providers
- 0461849: Fixed a bug where mastra.db file location was inconsistently created when running mastra dev vs running a file directly (tsx src/index.ts for ex)
- Updated dependencies [22643eb]
- Updated dependencies [6feb23f]
- Updated dependencies [f2d6727]
- Updated dependencies [301e4ee]
- Updated dependencies [dfbe4e9]
- Updated dependencies [9e81f35]
- Updated dependencies [caefaa2]
- Updated dependencies [c151ae6]
- Updated dependencies [52e0418]
- Updated dependencies [03236ec]
- Updated dependencies [3764e71]
- Updated dependencies [df982db]
- Updated dependencies [0461849]
- Updated dependencies [2259379]
- Updated dependencies [358f069]
  - @mastra/core@0.5.0-alpha.5

## 0.1.8-alpha.4

### Patch Changes

- 144b3d5: Update traces table UI, agent Chat UI
  Fix get workflows breaking
- Updated dependencies [d79aedf]
  - @mastra/core@0.5.0-alpha.4

## 0.1.8-alpha.3

### Patch Changes

- Updated dependencies [3d0e290]
  - @mastra/core@0.5.0-alpha.3

## 0.1.8-alpha.2

### Patch Changes

- Updated dependencies [02ffb7b]
  - @mastra/core@0.5.0-alpha.2

## 0.1.8-alpha.1

### Patch Changes

- Updated dependencies [dab255b]
  - @mastra/core@0.5.0-alpha.1

## 0.1.8-alpha.0

### Patch Changes

- c2dde91: Return full workflow details in api/workflows endpoint
- 731dd8a: Removed useless logging that showed up when user selected log drains tab on the playground
- 960690d: return runId from server on workflow watch
- Updated dependencies [59df7b6]
- Updated dependencies [29f3a82]
- Updated dependencies [59df7b6]
- Updated dependencies [c139344]
  - @mastra/core@0.5.0-alpha.0

## 0.1.7

### Patch Changes

- 30a4c29: fix mastra build errors related to esbuild not removing types
- e1e2705: Added --ignore-workspace when installing dependencies in mastra build with pnpm package manager
- Updated dependencies [1da20e7]
  - @mastra/core@0.4.4

## 0.1.7-alpha.0

### Patch Changes

- 30a4c29: fix mastra build errors related to esbuild not removing types
- e1e2705: Added --ignore-workspace when installing dependencies in mastra build with pnpm package manager
- Updated dependencies [1da20e7]
  - @mastra/core@0.4.4-alpha.0

## 0.1.6

### Patch Changes

- 80cdd76: Add hono routes for agent voice methods speakers, speak and listen
- 0fd78ac: Update vector store functions to use object params
- 0d25b75: Add all agent stream,generate option to cliend-js sdk
- bb4f447: Add support for commonjs
- Updated dependencies [0d185b1]
- Updated dependencies [ed55f1d]
- Updated dependencies [06aa827]
- Updated dependencies [0fd78ac]
- Updated dependencies [2512a93]
- Updated dependencies [e62de74]
- Updated dependencies [0d25b75]
- Updated dependencies [fd14a3f]
- Updated dependencies [8d13b14]
- Updated dependencies [3f369a2]
- Updated dependencies [3ee4831]
- Updated dependencies [4d4e1e1]
- Updated dependencies [bb4f447]
- Updated dependencies [108793c]
- Updated dependencies [5f28f44]
- Updated dependencies [dabecf4]
  - @mastra/core@0.4.3

## 0.1.6-alpha.4

### Patch Changes

- Updated dependencies [dabecf4]
  - @mastra/core@0.4.3-alpha.4

## 0.1.6-alpha.3

### Patch Changes

- 0fd78ac: Update vector store functions to use object params
- 0d25b75: Add all agent stream,generate option to cliend-js sdk
- bb4f447: Add support for commonjs
- Updated dependencies [0fd78ac]
- Updated dependencies [0d25b75]
- Updated dependencies [fd14a3f]
- Updated dependencies [3f369a2]
- Updated dependencies [4d4e1e1]
- Updated dependencies [bb4f447]
  - @mastra/core@0.4.3-alpha.3

## 0.1.6-alpha.2

### Patch Changes

- Updated dependencies [2512a93]
- Updated dependencies [e62de74]
  - @mastra/core@0.4.3-alpha.2

## 0.1.6-alpha.1

### Patch Changes

- 80cdd76: Add hono routes for agent voice methods speakers, speak and listen
- Updated dependencies [0d185b1]
- Updated dependencies [ed55f1d]
- Updated dependencies [8d13b14]
- Updated dependencies [3ee4831]
- Updated dependencies [108793c]
- Updated dependencies [5f28f44]
  - @mastra/core@0.4.3-alpha.1

## 0.1.6-alpha.0

### Patch Changes

- Updated dependencies [06aa827]
  - @mastra/core@0.4.3-alpha.0

## 0.1.5

### Patch Changes

- e4ee56c: Enable \* imports in analyze bundle
- 2d68431: Fix mastra server error processing
- e752340: Move storage/vector libSQL to own files so they do not get imported when not using bundlers.
- Updated dependencies [7fceae1]
- Updated dependencies [8d94c3e]
- Updated dependencies [99dcdb5]
- Updated dependencies [6cb63e0]
- Updated dependencies [f626fbb]
- Updated dependencies [e752340]
- Updated dependencies [eb91535]
  - @mastra/core@0.4.2

## 0.1.5-alpha.3

### Patch Changes

- e752340: Move storage/vector libSQL to own files so they do not get imported when not using bundlers.
- Updated dependencies [8d94c3e]
- Updated dependencies [99dcdb5]
- Updated dependencies [e752340]
- Updated dependencies [eb91535]
  - @mastra/core@0.4.2-alpha.2

## 0.1.5-alpha.2

### Patch Changes

- Updated dependencies [6cb63e0]
  - @mastra/core@0.4.2-alpha.1

## 0.1.5-alpha.1

### Patch Changes

- 2d68431: Fix mastra server error processing

## 0.1.5-alpha.0

### Patch Changes

- e4ee56c: Enable \* imports in analyze bundle
- Updated dependencies [7fceae1]
- Updated dependencies [f626fbb]
  - @mastra/core@0.4.2-alpha.0

## 0.1.4

### Patch Changes

- 967da43: Logger, transport fixes
- Updated dependencies [ce44b9b]
- Updated dependencies [967da43]
- Updated dependencies [b405f08]
  - @mastra/core@0.4.1

## 0.1.3

### Patch Changes

- 5297264: Fix build errors by changing contracts
- Updated dependencies [2fc618f]
- Updated dependencies [fe0fd01]
  - @mastra/core@0.4.0

## 0.1.3-alpha.1

### Patch Changes

- Updated dependencies [fe0fd01]
  - @mastra/core@0.4.0-alpha.1

## 0.1.3-alpha.0

### Patch Changes

- 5297264: Fix build errors by changing contracts
- Updated dependencies [2fc618f]
  - @mastra/core@0.4.0-alpha.0

## 0.1.2

### Patch Changes

- Updated dependencies [f205ede]
  - @mastra/core@0.3.0

## 0.1.1

### Patch Changes

- 936dc26: Add mastra server endpoints for watch/resume + plug watch and resume functionality to dev playground
- 91ef439: Add eslint and ran autofix
- aac1667: Improve treeshaking of core and output
- Updated dependencies [d59f1a8]
- Updated dependencies [91ef439]
- Updated dependencies [4a25be4]
- Updated dependencies [bf2e88f]
- Updated dependencies [2f0d707]
- Updated dependencies [aac1667]
  - @mastra/core@0.2.1

## 0.1.1-alpha.0

### Patch Changes

- 936dc26: Add mastra server endpoints for watch/resume + plug watch and resume functionality to dev playground
- 91ef439: Add eslint and ran autofix
- aac1667: Improve treeshaking of core and output
- Updated dependencies [d59f1a8]
- Updated dependencies [91ef439]
- Updated dependencies [4a25be4]
- Updated dependencies [bf2e88f]
- Updated dependencies [2f0d707]
- Updated dependencies [aac1667]
  - @mastra/core@0.2.1-alpha.0

## 0.1.0

### Minor Changes

- 4d4f6b6: Update deployer
- 5916f9d: Update deps from fixed to ^
- 8b416d9: Breaking changes

### Patch Changes

- 2ab57d6: Fix: Workflows require a trigger schema otherwise it fails to run in dev
- a1774e7: Improve bundling
- 291fe57: mastra openapi, swagger ui, dynamic servers
- e4d4ede: Better setLogger()
- 73d112c: Core and deployer fixes
- 9d1796d: Fix storage and eval serialization on api
- e27fe69: Add dir to deployer
- 246f06c: Fix import \* from telemetry package
- ac8c61a: Mastra server vector operations
- 82a6d53: better create-mastra tsconfig, better error for mastra server agent stream
- bdaf834: publish packages
- 7d83b92: Create default storage and move evals towards it
- 8fa48b9: Add an API to enhance agent instructions
- 685108a: Remove syncs and excess rag
- 5fdc87c: Update evals storage in attachListeners
- ae7bf94: Fix loggers messing up deploys
- b97ca96: Tracing into default storage
- ad2cd74: Deploy fix
- 7babd5c: CLI build and other
- a9b5ddf: Publish new versions
- 9066f95: CF deployer fixes
- 4139b43: Deployer utils
- ab01c53: Fix mastra server agent streamObject
- 1944807: Unified logger and major step in better logs
- 8aec8b7: Normalize imports to package name and dedupe while writing package.json after mastra build
- 685108a: Removing mastra syncs
- 382f4dc: move telemetry init to instrumentation.mjs file in build directory
- 7892533: Updated test evals to use Mastra Storage
- 9c10484: update all packages
- 88f18d7: Update cors support
- 70dabd9: Fix broken publish
- 1a41fbf: Fix playground workflow triggerData on execution
- 391d5ea: Add @opentelemetry/instrumentation to pkg json of build artifcat
- 8329f1a: Add debug env
- e6d8055: Added Mastra Storage to add and query live evals
- a18e96c: Array schemas for dev tool playground
- 5950de5: Added update instructions API
- b425845: Logger and execa logs
- 0696eeb: Cleanup Mastra server
- 6780223: fix workflow runId not unique per execution in dev
- a8a459a: Updated Evals table UI
- 0b96376: fix pino of being null
- cfb966f: Deprecate @mastra/tts for mastra speech providers
- 9625602: Use mastra core splitted bundles in other packages
- 72d1990: Updated evals table schema
- a291824: Deployer fixes
- 8ea426a: Fix patch
- c5f2d50: Split deployer package
- 7064554: deployer fixes
- 72c280b: Fixes
- b80ea8d: Fix bundling of server
- 42a2e69: Fix playground error parsing
- 28dceab: Catch apiKey error in dev
- a5604c4: Deployer initial
- 38b7f66: Update deployer logic
- b9c7047: Move to non deprecated table name for eval insertion
- 4a328af: Set request limit to 4.5MB
- 9ade36e: Changed measure for evals, added endpoints, attached metrics to agent, added ui for evals in playground, and updated docs
- d9c8dd0: Logger changes for default transports
- 9fb59d6: changeset
- f1e3105: Now that memory can be added to an agent, the playground needs to look up memory on the agent, not on mastra. Now the playground looks up on the agent to properly access memory
- ae7bf94: Changeset
- 4f1d1a1: Enforce types ann cleanup package.json
- Updated dependencies [f537e33]
- Updated dependencies [6f2c0f5]
- Updated dependencies [e4d4ede]
- Updated dependencies [0be7181]
- Updated dependencies [dd6d87f]
- Updated dependencies [9029796]
- Updated dependencies [6fa4bd2]
- Updated dependencies [f031a1f]
- Updated dependencies [8151f44]
- Updated dependencies [d7d465a]
- Updated dependencies [4d4f6b6]
- Updated dependencies [73d112c]
- Updated dependencies [592e3cf]
- Updated dependencies [9d1796d]
- Updated dependencies [e897f1c]
- Updated dependencies [4a54c82]
- Updated dependencies [3967e69]
- Updated dependencies [8ae2bbc]
- Updated dependencies [e9d1b47]
- Updated dependencies [016493a]
- Updated dependencies [bc40916]
- Updated dependencies [93a3719]
- Updated dependencies [7d83b92]
- Updated dependencies [9fb3039]
- Updated dependencies [d5e12de]
- Updated dependencies [e1dd94a]
- Updated dependencies [07c069d]
- Updated dependencies [5cdfb88]
- Updated dependencies [837a288]
- Updated dependencies [685108a]
- Updated dependencies [c8ff2f5]
- Updated dependencies [5fdc87c]
- Updated dependencies [ae7bf94]
- Updated dependencies [8e7814f]
- Updated dependencies [66a03ec]
- Updated dependencies [7d87a15]
- Updated dependencies [b97ca96]
- Updated dependencies [23dcb23]
- Updated dependencies [033eda6]
- Updated dependencies [8105fae]
- Updated dependencies [e097800]
- Updated dependencies [1944807]
- Updated dependencies [30322ce]
- Updated dependencies [1874f40]
- Updated dependencies [685108a]
- Updated dependencies [f7d1131]
- Updated dependencies [79acad0]
- Updated dependencies [7a19083]
- Updated dependencies [382f4dc]
- Updated dependencies [1ebd071]
- Updated dependencies [0b74006]
- Updated dependencies [2f17a5f]
- Updated dependencies [f368477]
- Updated dependencies [7892533]
- Updated dependencies [9c10484]
- Updated dependencies [b726bf5]
- Updated dependencies [70dabd9]
- Updated dependencies [21fe536]
- Updated dependencies [176bc42]
- Updated dependencies [401a4d9]
- Updated dependencies [2e099d2]
- Updated dependencies [0b826f6]
- Updated dependencies [d68b532]
- Updated dependencies [75bf3f0]
- Updated dependencies [e6d8055]
- Updated dependencies [e2e76de]
- Updated dependencies [ccbc581]
- Updated dependencies [5950de5]
- Updated dependencies [fe3dcb0]
- Updated dependencies [78eec7c]
- Updated dependencies [a8a459a]
- Updated dependencies [0be7181]
- Updated dependencies [7b87567]
- Updated dependencies [b524c22]
- Updated dependencies [d7d465a]
- Updated dependencies [df843d3]
- Updated dependencies [4534e77]
- Updated dependencies [d6d8159]
- Updated dependencies [0bd142c]
- Updated dependencies [9625602]
- Updated dependencies [72d1990]
- Updated dependencies [f6ba259]
- Updated dependencies [2712098]
- Updated dependencies [eedb829]
- Updated dependencies [5285356]
- Updated dependencies [74b3078]
- Updated dependencies [cb290ee]
- Updated dependencies [b4d7416]
- Updated dependencies [e608d8c]
- Updated dependencies [06b2c0a]
- Updated dependencies [002d6d8]
- Updated dependencies [e448a26]
- Updated dependencies [8b416d9]
- Updated dependencies [fd494a3]
- Updated dependencies [dc90663]
- Updated dependencies [c872875]
- Updated dependencies [3c4488b]
- Updated dependencies [a7b016d]
- Updated dependencies [fd75f3c]
- Updated dependencies [7f24c29]
- Updated dependencies [2017553]
- Updated dependencies [a10b7a3]
- Updated dependencies [cf6d825]
- Updated dependencies [963c15a]
- Updated dependencies [7365b6c]
- Updated dependencies [5ee67d3]
- Updated dependencies [d38f7a6]
- Updated dependencies [38b7f66]
- Updated dependencies [2fa7f53]
- Updated dependencies [1420ae2]
- Updated dependencies [f6da688]
- Updated dependencies [3700be1]
- Updated dependencies [9ade36e]
- Updated dependencies [10870bc]
- Updated dependencies [2b01511]
- Updated dependencies [a870123]
- Updated dependencies [ccf115c]
- Updated dependencies [04434b6]
- Updated dependencies [5811de6]
- Updated dependencies [9f3ab05]
- Updated dependencies [66a5392]
- Updated dependencies [4b1ce2c]
- Updated dependencies [14064f2]
- Updated dependencies [f5dfa20]
- Updated dependencies [327ece7]
- Updated dependencies [da2e8d3]
- Updated dependencies [95a4697]
- Updated dependencies [d5fccfb]
- Updated dependencies [3427b95]
- Updated dependencies [538a136]
- Updated dependencies [e66643a]
- Updated dependencies [b5393f1]
- Updated dependencies [d2cd535]
- Updated dependencies [c2dd6b5]
- Updated dependencies [67637ba]
- Updated dependencies [836f4e3]
- Updated dependencies [5ee2e78]
- Updated dependencies [cd02c56]
- Updated dependencies [01502b0]
- Updated dependencies [16e5b04]
- Updated dependencies [d9c8dd0]
- Updated dependencies [9fb59d6]
- Updated dependencies [a9345f9]
- Updated dependencies [99f1847]
- Updated dependencies [04f3171]
- Updated dependencies [8769a62]
- Updated dependencies [d5ec619]
- Updated dependencies [27275c9]
- Updated dependencies [ae7bf94]
- Updated dependencies [4f1d1a1]
- Updated dependencies [ee4de15]
- Updated dependencies [202d404]
- Updated dependencies [a221426]
  - @mastra/core@0.2.0

## 0.1.0-alpha.63

### Patch Changes

- 391d5ea: Add @opentelemetry/instrumentation to pkg json of build artifcat

## 0.1.0-alpha.62

### Patch Changes

- 382f4dc: move telemetry init to instrumentation.mjs file in build directory
- Updated dependencies [016493a]
- Updated dependencies [382f4dc]
- Updated dependencies [176bc42]
- Updated dependencies [d68b532]
- Updated dependencies [fe3dcb0]
- Updated dependencies [e448a26]
- Updated dependencies [fd75f3c]
- Updated dependencies [ccf115c]
- Updated dependencies [a221426]
  - @mastra/core@0.2.0-alpha.110

## 0.1.0-alpha.61

### Patch Changes

- b9c7047: Move to non deprecated table name for eval insertion

## 0.1.0-alpha.60

### Patch Changes

- Updated dependencies [d5fccfb]
  - @mastra/core@0.2.0-alpha.109

## 0.1.0-alpha.59

### Patch Changes

- Updated dependencies [5ee67d3]
- Updated dependencies [95a4697]
  - @mastra/core@0.2.0-alpha.108

## 0.1.0-alpha.58

### Patch Changes

- 8fa48b9: Add an API to enhance agent instructions
- Updated dependencies [66a5392]
  - @mastra/core@0.2.0-alpha.107

## 0.1.0-alpha.57

### Patch Changes

- a8a459a: Updated Evals table UI
- 4a328af: Set request limit to 4.5MB
- Updated dependencies [6f2c0f5]
- Updated dependencies [a8a459a]
  - @mastra/core@0.2.0-alpha.106

## 0.1.0-alpha.56

### Patch Changes

- 246f06c: Fix import \* from telemetry package

## 0.1.0-alpha.55

### Patch Changes

- Updated dependencies [1420ae2]
- Updated dependencies [99f1847]
  - @mastra/core@0.2.0-alpha.105

## 0.1.0-alpha.54

### Patch Changes

- 5fdc87c: Update evals storage in attachListeners
- b97ca96: Tracing into default storage
- 6780223: fix workflow runId not unique per execution in dev
- 72d1990: Updated evals table schema
- Updated dependencies [5fdc87c]
- Updated dependencies [b97ca96]
- Updated dependencies [72d1990]
- Updated dependencies [cf6d825]
- Updated dependencies [10870bc]
  - @mastra/core@0.2.0-alpha.104

## 0.1.0-alpha.53

### Patch Changes

- Updated dependencies [4534e77]
  - @mastra/core@0.2.0-alpha.103

## 0.1.0-alpha.52

### Patch Changes

- Updated dependencies [a9345f9]
  - @mastra/core@0.2.0-alpha.102

## 0.1.0-alpha.51

### Patch Changes

- 4f1d1a1: Enforce types ann cleanup package.json
- Updated dependencies [66a03ec]
- Updated dependencies [4f1d1a1]
  - @mastra/core@0.2.0-alpha.101

## 0.1.0-alpha.50

### Patch Changes

- 9d1796d: Fix storage and eval serialization on api
- Updated dependencies [9d1796d]
  - @mastra/core@0.2.0-alpha.100

## 0.1.0-alpha.49

### Patch Changes

- 7d83b92: Create default storage and move evals towards it
- Updated dependencies [7d83b92]
  - @mastra/core@0.2.0-alpha.99

## 0.1.0-alpha.48

### Patch Changes

- 8aec8b7: Normalize imports to package name and dedupe while writing package.json after mastra build

## 0.1.0-alpha.47

### Patch Changes

- 70dabd9: Fix broken publish
- Updated dependencies [70dabd9]
- Updated dependencies [202d404]
  - @mastra/core@0.2.0-alpha.98

## 0.1.0-alpha.46

### Patch Changes

- 7892533: Updated test evals to use Mastra Storage
- e6d8055: Added Mastra Storage to add and query live evals
- a18e96c: Array schemas for dev tool playground
- 5950de5: Added update instructions API
- f1e3105: Now that memory can be added to an agent, the playground needs to look up memory on the agent, not on mastra. Now the playground looks up on the agent to properly access memory
- Updated dependencies [07c069d]
- Updated dependencies [7892533]
- Updated dependencies [e6d8055]
- Updated dependencies [5950de5]
- Updated dependencies [df843d3]
- Updated dependencies [a870123]
  - @mastra/core@0.2.0-alpha.97

## 0.1.0-alpha.45

### Patch Changes

- Updated dependencies [74b3078]
  - @mastra/core@0.2.0-alpha.96

## 0.1.0-alpha.44

### Patch Changes

- 9fb59d6: changeset
- Updated dependencies [9fb59d6]
  - @mastra/core@0.2.0-alpha.95

## 0.1.0-alpha.43

### Minor Changes

- 8b416d9: Breaking changes

### Patch Changes

- 9c10484: update all packages
- Updated dependencies [9c10484]
- Updated dependencies [8b416d9]
  - @mastra/core@0.2.0-alpha.94

## 0.1.0-alpha.42

### Patch Changes

- 42a2e69: Fix playground error parsing
- Updated dependencies [5285356]
  - @mastra/core@0.2.0-alpha.93

## 0.1.0-alpha.41

### Patch Changes

- 0b96376: fix pino of being null

## 0.1.0-alpha.40

### Patch Changes

- 8329f1a: Add debug env

## 0.1.0-alpha.39

### Patch Changes

- 8ea426a: Fix patch

## 0.1.0-alpha.34

### Patch Changes

- b80ea8d: Fix bundling of server

## 0.1.0-alpha.38

### Minor Changes

- 4d4f6b6: Update deployer

### Patch Changes

- Updated dependencies [4d4f6b6]
  - @mastra/core@0.2.0-alpha.92

## 0.1.0-alpha.37

### Patch Changes

- Updated dependencies [d7d465a]
- Updated dependencies [d7d465a]
- Updated dependencies [2017553]
- Updated dependencies [a10b7a3]
- Updated dependencies [16e5b04]
  - @mastra/core@0.2.0-alpha.91

## 0.1.0-alpha.36

### Patch Changes

- 82a6d53: better create-mastra tsconfig, better error for mastra server agent stream
- Updated dependencies [8151f44]
- Updated dependencies [e897f1c]
- Updated dependencies [3700be1]
  - @mastra/core@0.2.0-alpha.90

## 0.1.0-alpha.35

### Patch Changes

- Updated dependencies [27275c9]
  - @mastra/core@0.2.0-alpha.89

## 0.1.0-alpha.34

### Patch Changes

- ab01c53: Fix mastra server agent streamObject
- Updated dependencies [ccbc581]
  - @mastra/core@0.2.0-alpha.88

## 0.1.0-alpha.33

### Patch Changes

- Updated dependencies [7365b6c]
  - @mastra/core@0.2.0-alpha.87

## 0.1.0-alpha.32

### Minor Changes

- 5916f9d: Update deps from fixed to ^

### Patch Changes

- Updated dependencies [6fa4bd2]
- Updated dependencies [e2e76de]
- Updated dependencies [7f24c29]
- Updated dependencies [67637ba]
- Updated dependencies [04f3171]
  - @mastra/core@0.2.0-alpha.86

## 0.0.1-alpha.31

### Patch Changes

- c5f2d50: Split deployer package
- Updated dependencies [e9d1b47]
  - @mastra/core@0.2.0-alpha.85

## 0.0.1-alpha.30

### Patch Changes

- e27fe69: Add dir to deployer

## 0.0.1-alpha.29

### Patch Changes

- 0696eeb: Cleanup Mastra server
- 38b7f66: Update deployer logic
- Updated dependencies [2f17a5f]
- Updated dependencies [cb290ee]
- Updated dependencies [b4d7416]
- Updated dependencies [38b7f66]
  - @mastra/core@0.2.0-alpha.84

## 0.0.1-alpha.28

### Patch Changes

- 2ab57d6: Fix: Workflows require a trigger schema otherwise it fails to run in dev
- 9625602: Use mastra core splitted bundles in other packages
- Updated dependencies [30322ce]
- Updated dependencies [78eec7c]
- Updated dependencies [9625602]
- Updated dependencies [8769a62]
  - @mastra/core@0.2.0-alpha.83

## 0.0.1-alpha.27

### Patch Changes

- 73d112c: Core and deployer fixes
- ac8c61a: Mastra server vector operations
- Updated dependencies [73d112c]
  - @mastra/core@0.1.27-alpha.82

## 0.0.1-alpha.26

### Patch Changes

- Updated dependencies [9fb3039]
  - @mastra/core@0.1.27-alpha.81

## 0.0.1-alpha.25

### Patch Changes

- Updated dependencies [327ece7]
  - @mastra/core@0.1.27-alpha.80

## 0.0.1-alpha.24

### Patch Changes

- Updated dependencies [21fe536]
  - @mastra/core@0.1.27-alpha.79

## 0.0.1-alpha.23

### Patch Changes

- 88f18d7: Update cors support

## 0.0.1-alpha.22

### Patch Changes

- 685108a: Remove syncs and excess rag
- 685108a: Removing mastra syncs
- Updated dependencies [685108a]
- Updated dependencies [685108a]
  - @mastra/core@0.1.27-alpha.78

## 0.0.1-alpha.21

### Patch Changes

- cfb966f: Deprecate @mastra/tts for mastra speech providers
- Updated dependencies [8105fae]
  - @mastra/core@0.1.27-alpha.77

## 0.0.1-alpha.20

### Patch Changes

- ae7bf94: Fix loggers messing up deploys
- ae7bf94: Changeset
- Updated dependencies [ae7bf94]
- Updated dependencies [ae7bf94]
  - @mastra/core@0.1.27-alpha.76

## 0.0.1-alpha.19

### Patch Changes

- 7064554: deployer fixes
- Updated dependencies [23dcb23]
  - @mastra/core@0.1.27-alpha.75

## 0.0.1-alpha.18

### Patch Changes

- Updated dependencies [7b87567]
  - @mastra/core@0.1.27-alpha.74

## 0.0.1-alpha.17

### Patch Changes

- Updated dependencies [3427b95]
  - @mastra/core@0.1.27-alpha.73

## 0.0.1-alpha.16

### Patch Changes

- e4d4ede: Better setLogger()
- Updated dependencies [e4d4ede]
- Updated dependencies [06b2c0a]
  - @mastra/core@0.1.27-alpha.72

## 0.0.1-alpha.15

### Patch Changes

- d9c8dd0: Logger changes for default transports
- Updated dependencies [d9c8dd0]
  - @mastra/core@0.1.27-alpha.71

## 0.0.1-alpha.14

### Patch Changes

- ad2cd74: Deploy fix

## 0.0.1-alpha.13

### Patch Changes

- a1774e7: Improve bundling

## 0.0.1-alpha.12

### Patch Changes

- 28dceab: Catch apiKey error in dev

## 0.0.1-alpha.11

### Patch Changes

- bdaf834: publish packages

## 0.0.1-alpha.10

### Patch Changes

- Updated dependencies [dd6d87f]
- Updated dependencies [04434b6]
  - @mastra/core@0.1.27-alpha.70

## 0.0.1-alpha.9

### Patch Changes

- 9066f95: CF deployer fixes

## 0.0.1-alpha.8

### Patch Changes

- b425845: Logger and execa logs

## 0.0.1-alpha.7

### Patch Changes

- 1944807: Unified logger and major step in better logs
- 9ade36e: Changed measure for evals, added endpoints, attached metrics to agent, added ui for evals in playground, and updated docs
- Updated dependencies [1944807]
- Updated dependencies [9ade36e]
  - @mastra/core@0.1.27-alpha.69

## 0.0.1-alpha.6

### Patch Changes

- 291fe57: mastra openapi, swagger ui, dynamic servers
- 1a41fbf: Fix playground workflow triggerData on execution

## 0.0.1-alpha.5

### Patch Changes

- Updated dependencies [0be7181]
- Updated dependencies [0be7181]
  - @mastra/core@0.1.27-alpha.68

## 0.0.1-alpha.4

### Patch Changes

- 7babd5c: CLI build and other

## 0.0.1-alpha.3

### Patch Changes

- a291824: Deployer fixes
- Updated dependencies [c8ff2f5]
  - @mastra/core@0.1.27-alpha.67

## 0.0.1-alpha.2

### Patch Changes

- a9b5ddf: Publish new versions
- 72c280b: Fixes

## 0.0.1-alpha.0

### Patch Changes

- 4139b43: Deployer utils
- a5604c4: Deployer initial
