import * as fs from 'fs-extra';
import klawSync from 'klaw-sync';
import * as path from 'path';
import {pack} from 'tar-stream';
import * as zlib from 'zlib';

// build path
const ROOT_FOLDER = path.join(__dirname, '..');
const BIN_FOLDER = path.join(ROOT_FOLDER, 'bin');
const PREBUILDS_FOLDER = path.join(ROOT_FOLDER, 'prebuilds');

// start pack
const tarName = `${process.env.npm_package_name}-v${process.env.npm_package_version}-napi-v3-${process.platform}-${
    process.arch}.tar.gz`;
const tarPath = path.join(PREBUILDS_FOLDER, tarName);

const tarStream = pack();
fs.ensureDirSync(PREBUILDS_FOLDER);
const ws = fs.createWriteStream(tarPath);

tarStream.pipe(zlib.createGzip({level: 9})).pipe(ws);

// enumerate all files under BIN folder
const entries = klawSync(BIN_FOLDER, {nodir: true}).map(i => ({
                                                          path: i.path,
                                                          name: path.relative(ROOT_FOLDER, i.path).replace(/\\/g, '/'),
                                                          size: i.stats.size,
                                                          mode: i.stats.mode | parseInt('444', 8) | parseInt('222', 8),
                                                          gid: i.stats.gid,
                                                          uid: i.stats.uid
                                                        }));

console.log(`=== start to pack prebuild: ${tarName}`);

packNextFile();

function packNextFile(): void {
  const nextEntry = entries.shift();
  if (nextEntry) {
    console.log(`  adding file: ${nextEntry.name}`);
    const stream = tarStream.entry(nextEntry);
    fs.createReadStream(nextEntry.path).pipe(stream).on('finish', packNextFile);
  } else {
    console.log('=== finished packing prebuild.');
    tarStream.finalize();
  }
}
