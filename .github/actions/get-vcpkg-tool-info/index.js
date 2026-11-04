const fs = require("fs");
const path = require("path");

const infoPath = path.join(__dirname, "../../../tools/ci_build/vcpkg_tool_info.json");
const { release_tag: releaseTag, sha512 } = JSON.parse(fs.readFileSync(infoPath, "utf8"));

fs.appendFileSync(process.env.GITHUB_OUTPUT, `release_tag=${releaseTag}\nsha512=${sha512}\n`);
