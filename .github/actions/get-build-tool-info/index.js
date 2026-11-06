const fs = require("fs");
const path = require("path");

function getToolInfo(toolInfo, toolName, hostOS = process.platform, hostArch = process.arch) {
  const info = toolInfo[toolName];
  if (!info) {
    throw new Error(`Unknown build tool '${toolName}'`);
  }

  const { version } = info;
  let { sha512 } = info;

  if (sha512 && typeof sha512 === "object") {
    const osKey = {
      win32: "windows",
      linux: "linux",
      darwin: "macos",
    }[hostOS];

    const archKey = {
      amd64: "x64",
      x86_64: "x64",
      x64: "x64",
      arm64: "arm64",
      aarch64: "arm64",
    }[hostArch.toLowerCase()];

    const osHashes = sha512[osKey];

    sha512 = osHashes && archKey && (osHashes.universal ?? osHashes[archKey]);
    if (!sha512) {
      throw new Error(`No ${toolName} SHA-512 configured for host OS '${hostOS}' and architecture '${hostArch}'`);
    }
  }

  if (typeof version !== "string" || version.length === 0 || /[^A-Za-z0-9._+-]/.test(version)) {
    throw new Error(`Invalid ${toolName} version`);
  }

  if (typeof sha512 !== "string" || sha512.length !== 128 || /[^0-9a-fA-F]/.test(sha512)) {
    throw new Error(`Invalid ${toolName} SHA-512`);
  }

  return { version, sha512 };
}

function main() {
  const toolInfoPath = path.join(__dirname, "..", "..", "..", "tools", "ci_build", "build_tool_info.json");
  const toolInfo = JSON.parse(fs.readFileSync(toolInfoPath, "utf8"));

  const toolNames = (process.env.INPUT_TOOLS ?? "").split(",").map((toolName) => toolName.trim());

  const outputs = toolNames.flatMap((toolName) => {
    const { version, sha512 } = getToolInfo(toolInfo, toolName);

    return [
      `${toolName}-version=${version}`,
      `${toolName}-sha512=${sha512}`,
    ];
  });

  fs.appendFileSync(process.env.GITHUB_OUTPUT, outputs.join("\n") + "\n");
}

module.exports = { getToolInfo };

if (require.main === module) {
  main();
}
