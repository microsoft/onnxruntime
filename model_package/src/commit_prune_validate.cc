// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file commit_prune_validate.cc
/// \brief Commit, prune, and validate implementation.

#include "model_package.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "asset_hasher.h"
#include "manifest_parser.h"
#include "model_package_impl.h"
#include "path_resolver.h"
#include "status_impl.h"

namespace fs = std::filesystem;
namespace mp = model_package;
using model_package::MakeStatus;
using nlohmann::ordered_json;

namespace {

ModelPackageStatus* NullArg(const char* name) {
  return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                    std::string("model_package: '") + name + "' must not be null.");
}

// ─────────────────────────────────────────────────────────────────────────────
// fsync / random helpers (POSIX). Windows would substitute FlushFileBuffers +
// BCryptGenRandom; deferred to a follow-up.
// ─────────────────────────────────────────────────────────────────────────────

std::string RandomSuffix() {
  std::random_device rd;
  uint64_t hi = (uint64_t(rd()) << 32) | rd();
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hi));
  return buf;
}

ModelPackageStatus* FsyncPath(const fs::path& p, bool is_dir) {
#ifdef _WIN32
  (void)p;
  (void)is_dir;
  return nullptr;
#else
  int flags = is_dir ? (O_RDONLY | O_DIRECTORY) : O_RDONLY;
  int fd = ::open(p.c_str(), flags);
  if (fd < 0) {
    // Best-effort: missing fsync targets are not fatal on tmpfs etc.
    return nullptr;
  }
  if (::fsync(fd) != 0) {
    int err = errno;
    ::close(fd);
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      std::string("fsync '") + p.string() + "' failed: " + std::strerror(err));
  }
  ::close(fd);
  return nullptr;
#endif
}

ModelPackageStatus* WriteFileAtomic(const fs::path& final_path, const std::string& bytes) {
  fs::path tmp = final_path;
  tmp += ".tmp." + RandomSuffix();
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "Cannot open '" + tmp.string() + "' for writing.");
    }
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!f) {
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "Write to '" + tmp.string() + "' failed.");
    }
  }
  if (auto* s = FsyncPath(tmp, /*is_dir=*/false)) return s;
  std::error_code ec;
  fs::rename(tmp, final_path, ec);
  if (ec) {
    fs::remove(tmp, ec);
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      "Rename '" + tmp.string() + "' -> '" + final_path.string() +
                          "' failed: " + ec.message());
  }
  if (auto* s = FsyncPath(final_path.parent_path(), /*is_dir=*/true)) return s;
  return nullptr;
}

ModelPackageStatus* CopyTreeNoFollow(const fs::path& src, const fs::path& dst) {
  // Recursively copy `src` into `dst`. Refuses to follow symlinks (consistent
  // with the directory hash semantics) so the on-disk bytes match the URI we
  // already computed.
  std::error_code ec;
  fs::create_directories(dst, ec);
  if (ec) return MakeStatus(MODEL_PACKAGE_ERR_IO,
                            "mkdir '" + dst.string() + "': " + ec.message());
  for (fs::recursive_directory_iterator it(src, fs::directory_options::none, ec), end;
       it != end; it.increment(ec)) {
    if (ec) return MakeStatus(MODEL_PACKAGE_ERR_IO,
                              "iterate '" + src.string() + "': " + ec.message());
    const auto& entry = *it;
    fs::path rel = fs::relative(entry.path(), src, ec);
    fs::path target = dst / rel;
    if (entry.is_symlink()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "shared asset source contains a symlink: '" + entry.path().string() + "'.");
    }
    if (entry.is_directory()) {
      fs::create_directories(target, ec);
      if (ec) return MakeStatus(MODEL_PACKAGE_ERR_IO,
                                "mkdir '" + target.string() + "': " + ec.message());
    } else if (entry.is_regular_file()) {
      fs::create_directories(target.parent_path(), ec);
      fs::copy_file(entry.path(), target, fs::copy_options::overwrite_existing, ec);
      if (ec) return MakeStatus(MODEL_PACKAGE_ERR_IO,
                                "copy '" + entry.path().string() + "' -> '" +
                                    target.string() + "': " + ec.message());
      if (auto* s = FsyncPath(target, /*is_dir=*/false)) return s;
    } else {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "unsupported file kind in shared asset source: '" +
                            entry.path().string() + "'.");
    }
  }
  if (auto* s = FsyncPath(dst, /*is_dir=*/true)) return s;
  return nullptr;
}

ModelPackageStatus* CheckPortableConfinement(const fs::path& root,
                                             const fs::path& candidate,
                                             const std::string& where) {
  std::error_code ec;
  fs::path c = candidate.lexically_normal();
  fs::path r = root.lexically_normal();
  if (c.is_absolute()) {
    // Confirm c is under r.
    auto rel = fs::relative(c, r, ec);
    // An empty relative path, or one whose first component is "..", escapes the root.
    // (Checking only the first character would wrongly reject in-root dot-prefixed names
    // such as ".hidden/component.json".)
    if (ec || rel.empty() || rel.begin()->string() == "..") {
      return MakeStatus(MODEL_PACKAGE_ERR_PATH_CONFINEMENT,
                        where + ": absolute path '" + c.string() +
                            "' escapes package_root '" + r.string() + "' (portable layout).");
    }
  } else {
    // Relative: a leading ".." escapes.
    auto first = c.begin();
    if (first != c.end() && first->string() == "..") {
      return MakeStatus(MODEL_PACKAGE_ERR_PATH_CONFINEMENT,
                        where + ": relative path '" + c.string() +
                            "' escapes package_root (portable layout).");
    }
  }
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Manifest serialization
// ─────────────────────────────────────────────────────────────────────────────

std::string SerializeManifestForCommit(const ModelPackage* pkg) {
  // Use the live in-memory manifest, but for external components, the
  // ComponentRecord::body may have diverged from the string path. The manifest
  // entry stays as the string in that case; the body is serialized separately
  // into the external file.
  return pkg->manifest.dump(2) + "\n";
}

ordered_json SerializeComponentBody(const mp::ComponentRecord* comp) {
  return comp->body;
}

// ─────────────────────────────────────────────────────────────────────────────
// In-place commit (PRESERVE / DENSE)
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* CheckDenseConstraints(ModelPackage* pkg) {
  // Reject external executor_info in dense mode (dense flattens everything,
  // but the in-memory model never loads external executor_info bodies, so we
  // can't inline them surgically. ERR_STATE so the caller's intent is clear.)
  for (const auto& comp : pkg->components) {
    auto vit = comp->body.find("variants");
    if (vit == comp->body.end() || !vit->is_object()) continue;
    for (auto v = vit->begin(); v != vit->end(); ++v) {
      auto ei = v->find("executor_info");
      if (ei == v->end() || !ei->is_object()) continue;
      for (auto e = ei->begin(); e != ei->end(); ++e) {
        if (e->is_string()) {
          return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                            "WRITE_DENSE: component '" + comp->name + "' variant '" +
                                v.key() + "' has external executor_info '" + e.key() +
                                "' (string path). Convert to inline via "
                                "SetVariantExecutorInfoInline before dense commit.");
        }
      }
    }
  }
  return nullptr;
}

ModelPackageStatus* CommitSharedAssetsCopyIn(ModelPackage* pkg, const fs::path& root) {
  if (pkg->pending_shared_asset_copies.empty()) return nullptr;
  fs::path assets_root = root / "shared_assets";
  std::error_code ec;
  fs::create_directories(assets_root, ec);
  for (const auto& [uri, src] : pkg->pending_shared_asset_copies) {
    std::string dir_name = mp::DefaultSharedAssetDirName(uri);
    fs::path final_dir = assets_root / dir_name;
    if (fs::exists(final_dir, ec)) continue;  // already materialized — trust it.
    fs::path stage_dir = assets_root / (dir_name + ".tmp." + RandomSuffix());
    if (auto* s = CopyTreeNoFollow(src, stage_dir)) {
      fs::remove_all(stage_dir, ec);
      return s;
    }
    // Re-hash staging to verify TOCTOU did not strike.
    std::string verify_uri;
    if (auto* s = mp::ComputeDirectoryAssetUri(stage_dir, &verify_uri)) {
      fs::remove_all(stage_dir, ec);
      return s;
    }
    if (verify_uri != uri) {
      fs::remove_all(stage_dir, ec);
      return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                        "Shared asset source mutated during commit: expected " +
                            uri + ", staged " + verify_uri + ".");
    }
    fs::rename(stage_dir, final_dir, ec);
    if (ec) {
      fs::remove_all(stage_dir, ec);
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "Rename shared asset dir '" + stage_dir.string() + "' -> '" +
                            final_dir.string() + "' failed: " + ec.message());
    }
    if (auto* s = FsyncPath(assets_root, /*is_dir=*/true)) return s;
  }
  return nullptr;
}

ModelPackageStatus* CommitExternalComponents(ModelPackage* pkg) {
  // Write each external component's current in-memory body to its disk file.
  // These are library-owned; for in-place PRESERVE commit we re-emit them
  // every time (cheaper than tracking dirtiness). External executor_info
  // files are opaque and intentionally left untouched.
  for (const auto& comp : pkg->components) {
    if (comp->storage != mp::ComponentStorage::kExternal) continue;
    fs::path path = comp->external_path;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::string text = SerializeComponentBody(comp.get()).dump(2) + "\n";
    if (auto* s = WriteFileAtomic(path, text)) return s;
  }
  return nullptr;
}

ModelPackageStatus* CommitInPlace(ModelPackage* pkg, ModelPackageWriteMode mode) {
  if (pkg->package_root.empty()) {
    return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                      "Commit: package has no package_root. Use dest_root variant.");
  }
  std::error_code ec;
  if (!fs::is_directory(pkg->package_root, ec)) {
    fs::create_directories(pkg->package_root, ec);
    if (ec) {
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "Cannot create package_root '" + pkg->package_root.string() +
                            "': " + ec.message());
    }
  }

  // Portable confinement pre-flight for external paths.
  if (pkg->layout == "portable") {
    for (const auto& comp : pkg->components) {
      if (comp->storage == mp::ComponentStorage::kExternal) {
        if (auto* s = CheckPortableConfinement(pkg->package_root, comp->external_path,
                                               "component '" + comp->name + "'")) return s;
      }
    }
  }

  // Dense mode: flatten external components into manifest before writing.
  if (mode == MODEL_PACKAGE_WRITE_DENSE) {
    if (auto* s = CheckDenseConstraints(pkg)) return s;
    for (auto& comp : pkg->components) {
      if (comp->storage == mp::ComponentStorage::kExternal) {
        pkg->manifest["components"][comp->name] = comp->body;
        // After commit, this becomes inline.
        comp->storage = mp::ComponentStorage::kInline;
        comp->external_path.clear();
        comp->component_dir = pkg->package_root;
      }
    }
  }

  if (auto* s = CommitSharedAssetsCopyIn(pkg, pkg->package_root)) return s;
  if (mode != MODEL_PACKAGE_WRITE_DENSE) {
    if (auto* s = CommitExternalComponents(pkg)) return s;
  }

  // Final manifest write.
  fs::path manifest_path = pkg->package_root / "manifest.json";
  if (auto* s = WriteFileAtomic(manifest_path, SerializeManifestForCommit(pkg))) return s;

  pkg->pending_shared_asset_copies.clear();

  // Re-derive shared assets + info view to pick up the materialized assets.
  if (auto* s = mp::RefreshSharedAssets(pkg, mp::PathOptionsFor(pkg))) return s;
  if (auto* s = mp::RefreshPackageMetadata(pkg)) return s;
  mp::DropViewCache(pkg);
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// dest_root commit ("save as"): write to dest_root, then re-parse & swap.
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* CommitToDestRoot(ModelPackage* pkg,
                                     const fs::path& dest_root,
                                     ModelPackageWriteMode mode) {
  std::error_code ec;
  if (fs::exists(dest_root, ec)) {
    if (!fs::is_directory(dest_root, ec)) {
      return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                        "Commit dest_root '" + dest_root.string() + "' exists and is not a directory.");
    }
    if (!fs::is_empty(dest_root, ec)) {
      return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                        "Commit dest_root '" + dest_root.string() + "' is not empty.");
    }
  } else {
    fs::create_directories(dest_root, ec);
    if (ec) {
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "Cannot create dest_root '" + dest_root.string() + "': " + ec.message());
    }
  }

  // Build a snapshot manifest mirroring `pkg->manifest`, then handle assets.
  ordered_json manifest = pkg->manifest;

  // Dense mode constraints up-front.
  if (mode == MODEL_PACKAGE_WRITE_DENSE) {
    if (auto* s = CheckDenseConstraints(pkg)) return s;
    for (const auto& comp : pkg->components) {
      if (comp->storage == mp::ComponentStorage::kExternal) {
        manifest["components"][comp->name] = comp->body;
      }
    }
  }

  // Copy all shared assets into dest_root. Any manifest override entries are
  // re-mapped to the default convention path under dest_root.
  fs::path assets_root = dest_root / "shared_assets";
  // Gather source dirs for every URI we know about.
  // 1. URIs already on disk (under current package_root) and not in pending: copy from there.
  // 2. Pending copy_in sources: copy from staged source.
  // 3. Manifest override entries: copy from the override path.
  std::vector<std::pair<std::string, fs::path>> to_copy;
  for (const auto& rec : pkg->shared_assets) {
    auto pit = pkg->pending_shared_asset_copies.find(rec->uri);
    if (pit != pkg->pending_shared_asset_copies.end()) {
      to_copy.emplace_back(rec->uri, pit->second);
    } else {
      to_copy.emplace_back(rec->uri, rec->resolved_path);
    }
  }
  // Pending copies without a SharedAssetRecord shouldn't happen now that
  // LoadSharedAssets surfaces pending copies, but stay defensive.
  for (const auto& [uri, src] : pkg->pending_shared_asset_copies) {
    if (pkg->shared_asset_index_by_uri.find(uri) == pkg->shared_asset_index_by_uri.end()) {
      to_copy.emplace_back(uri, src);
    }
  }
  // Only materialize shared_assets/ when something will actually land in it.
  if (!to_copy.empty()) {
    fs::create_directories(assets_root, ec);
  }

  for (const auto& [uri, src] : to_copy) {
    if (!fs::is_directory(src, ec)) {
      return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                        "Commit dest_root: shared asset source '" + src.string() +
                            "' for " + uri + " is not a directory.");
    }
    std::string dir_name = mp::DefaultSharedAssetDirName(uri);
    fs::path final_dir = assets_root / dir_name;
    fs::path stage_dir = assets_root / (dir_name + ".tmp." + RandomSuffix());
    if (auto* s = CopyTreeNoFollow(src, stage_dir)) {
      fs::remove_all(stage_dir, ec);
      return s;
    }
    std::string verify_uri;
    if (auto* s = mp::ComputeDirectoryAssetUri(stage_dir, &verify_uri)) {
      fs::remove_all(stage_dir, ec);
      return s;
    }
    if (verify_uri != uri) {
      fs::remove_all(stage_dir, ec);
      return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                        "Shared asset hash mismatch during dest_root commit: expected " +
                            uri + ", staged " + verify_uri);
    }
    fs::rename(stage_dir, final_dir, ec);
    if (ec) {
      fs::remove_all(stage_dir, ec);
      return MakeStatus(MODEL_PACKAGE_ERR_IO, "Rename failed: " + ec.message());
    }
  }
  // All assets now live at the default convention path; drop overrides.
  manifest.erase("shared_assets");

  // External components (PRESERVE mode): re-emit under dest_root using the same
  // path string from the manifest. We treat the manifest string as relative to
  // dest_root for portable mode; absolute paths are kept as-is iff the layout
  // is installed.
  if (mode == MODEL_PACKAGE_WRITE_PRESERVE) {
    auto comps_it = manifest.find("components");
    if (comps_it != manifest.end() && comps_it->is_object()) {
      for (auto e = comps_it->begin(); e != comps_it->end(); ++e) {
        if (!e->is_string()) continue;
        fs::path p(e->get<std::string>());
        fs::path target;
        if (p.is_absolute()) {
          if (pkg->layout == "portable") {
            return MakeStatus(MODEL_PACKAGE_ERR_PATH_CONFINEMENT,
                              "dest_root commit (portable): component '" + e.key() +
                                  "' has absolute path '" + p.string() + "'.");
          }
          target = p;
        } else {
          target = dest_root / p;
          std::error_code ec2;
          fs::path normalized = target.lexically_normal();
          if (normalized.string().find(dest_root.lexically_normal().string()) != 0) {
            return MakeStatus(MODEL_PACKAGE_ERR_PATH_CONFINEMENT,
                              "dest_root commit (portable): component '" + e.key() +
                                  "' relative path '" + p.string() + "' escapes dest_root.");
          }
          target = normalized;
        }
        // Find the corresponding component body to write.
        std::string ext_body;
        for (const auto& comp : pkg->components) {
          if (comp->name == e.key()) {
            ext_body = comp->body.dump(2) + "\n";
            break;
          }
        }
        std::error_code ec_md;
        fs::create_directories(target.parent_path(), ec_md);
        if (auto* s = WriteFileAtomic(target, ext_body)) return s;
      }
    }
  }

  fs::path manifest_path = dest_root / "manifest.json";
  if (auto* s = WriteFileAtomic(manifest_path, manifest.dump(2) + "\n")) return s;

  // Re-parse the newly written package into a fresh state and swap in.
  ModelPackageOpenOptions opts{};
  opts.allow_external_paths = pkg->allow_external_paths;
  opts.follow_symlinks = pkg->follow_symlinks;
  opts.strict_unknown_fields = pkg->strict_unknown_fields;
  ModelPackage fresh;
  if (auto* s = mp::ParsePackage(dest_root, opts, &fresh)) {
    return s;
  }
  // Tear down the existing view cache for the old package, then swap.
  mp::DropViewCache(pkg);
  // Field-by-field swap (the opaque struct is non-trivial; std::swap of the
  // struct works because all members are move/swap-friendly).
  std::swap(pkg->package_root, fresh.package_root);
  std::swap(pkg->manifest, fresh.manifest);
  std::swap(pkg->layout, fresh.layout);
  std::swap(pkg->components, fresh.components);
  std::swap(pkg->shared_assets, fresh.shared_assets);
  std::swap(pkg->component_index_by_name, fresh.component_index_by_name);
  std::swap(pkg->shared_asset_index_by_uri, fresh.shared_asset_index_by_uri);
  std::swap(pkg->package_name_cache, fresh.package_name_cache);
  std::swap(pkg->package_version_cache, fresh.package_version_cache);
  std::swap(pkg->description_cache, fresh.description_cache);
  std::swap(pkg->layout_cache, fresh.layout_cache);
  std::swap(pkg->additional_metadata_cache, fresh.additional_metadata_cache);
  std::swap(pkg->schema_version_major, fresh.schema_version_major);
  std::swap(pkg->schema_version_minor, fresh.schema_version_minor);
  pkg->pending_shared_asset_copies.clear();
  pkg->info_cache.reset();

  if (auto* s = mp::RefreshPackageMetadata(pkg)) return s;
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Prune
// ─────────────────────────────────────────────────────────────────────────────

constexpr std::chrono::seconds kPruneGrace{60};

bool IsTmpName(const fs::path& p) {
  std::string name = p.filename().string();
  return name.find(".tmp.") != std::string::npos;
}

bool IsOldEnough(const fs::path& p) {
  std::error_code ec;
  auto last = fs::last_write_time(p, ec);
  if (ec) return false;
  auto now = decltype(last)::clock::now();
  return (now - last) >= kPruneGrace;
}

bool IsAncestorOrEqual(const fs::path& ancestor, const fs::path& descendant) {
  // ancestor == descendant, or descendant lives under ancestor (boundary aware).
  auto a = ancestor.lexically_normal().generic_string();
  auto d = descendant.lexically_normal().generic_string();
  if (d.size() < a.size()) return false;
  if (d.compare(0, a.size(), a) != 0) return false;
  return d.size() == a.size() || d[a.size()] == '/';
}

std::vector<fs::path> CollectLiveDirs(const ModelPackage* pkg) {
  std::vector<fs::path> out;
  for (const auto& c : pkg->components) {
    if (c->storage == mp::ComponentStorage::kExternal) {
      out.push_back(c->component_dir);
    }
    for (const auto& v : c->variants) {
      if (v->resolved_directory.has_value()) {
        out.push_back(*v->resolved_directory);
      }
    }
  }
  return out;
}

// Drop entries we've handled (removed, or unsafe to touch). Entries that
// reference live state stay for a future Prune call. Tracked orphans don't
// wait on the kPruneGrace window: they were recorded by an in-session
// mutation, so there's no concurrent writer to protect against. The grace
// window is still applied to the shared_assets sweep below, which discovers
// candidates fresh from disk.
void SweepOrphanDirs(ModelPackage* pkg,
                     std::vector<fs::path>* pending,
                     const std::vector<fs::path>& live_dirs) {
  pending->erase(std::remove_if(pending->begin(), pending->end(), [&](const fs::path& p) {
                   if (!mp::IsInsidePackageRoot(pkg, p)) return true;  // outside our scope
                   std::error_code ec;
                   if (!fs::exists(p, ec)) return true;
                   // Skip if any live dir IS p or lives under it; deleting would damage live state.
                   for (const auto& live : live_dirs) {
                     if (IsAncestorOrEqual(p, live)) return false;
                   }
                   fs::remove_all(p, ec);
                   return true;
                 }),
                 pending->end());
}

}  // namespace

namespace model_package {

bool IsInsidePackageRoot(const ModelPackage* pkg, const fs::path& p) {
  if (pkg->package_root.empty()) return false;
  return IsAncestorOrEqual(pkg->package_root, p);
}

void RecordOrphanVariantDir(ModelPackage* pkg, const VariantRecord& v) {
  if (!v.resolved_directory.has_value()) return;
  if (!IsInsidePackageRoot(pkg, *v.resolved_directory)) return;
  pkg->pending_orphan_variant_dirs.push_back(*v.resolved_directory);
}

void RecordOrphanComponent(ModelPackage* pkg, const ComponentRecord& c) {
  for (const auto& v : c.variants) {
    RecordOrphanVariantDir(pkg, *v);
  }
  if (c.storage == ComponentStorage::kExternal &&
      IsInsidePackageRoot(pkg, c.component_dir)) {
    pkg->pending_orphan_component_dirs.push_back(c.component_dir);
  }
}

}  // namespace model_package

extern "C" {

ModelPackageStatus* ModelPackage_Commit(ModelPackage* pkg,
                                        const char* dest_root_or_null,
                                        ModelPackageWriteMode mode) {
  if (!pkg) return NullArg("pkg");
  if (dest_root_or_null) {
    return CommitToDestRoot(pkg, fs::path(dest_root_or_null), mode);
  }
  return CommitInPlace(pkg, mode);
}

ModelPackageStatus* ModelPackage_Prune(ModelPackage* pkg) {
  if (!pkg) return NullArg("pkg");
  if (pkg->package_root.empty()) return nullptr;

  // Shared assets are NEVER auto-pruned. The library cannot prove an asset is
  // unused without parsing every consumer's executor_info payload, and a
  // mistaken delete is worse than disk bloat for content-addressed dirs that
  // dedupe naturally. Callers reclaim shared assets via explicit
  // ModelPackage_RemoveSharedAsset(uri) (which still requires consumer-aware
  // knowledge of what's reachable).
  //
  // Stale `.tmp.<suffix>` staging dirs from interrupted commits are reclaimed
  // here after a grace window: they belong to this library's own staging
  // protocol and aren't user data.
  std::error_code ec;
  fs::path assets_root = pkg->package_root / "shared_assets";
  if (fs::is_directory(assets_root, ec)) {
    for (const auto& entry : fs::directory_iterator(assets_root, ec)) {
      if (ec) break;
      if (!entry.is_directory()) continue;
      if (!IsTmpName(entry.path())) continue;
      if (!IsOldEnough(entry.path())) continue;
      fs::remove_all(entry.path(), ec);
    }
  }

  // Tracked-orphan sweep: components before variants so a component_dir
  // removal reclaims its child variant dirs in one shot.
  std::vector<fs::path> live_dirs = CollectLiveDirs(pkg);
  SweepOrphanDirs(pkg, &pkg->pending_orphan_component_dirs, live_dirs);
  SweepOrphanDirs(pkg, &pkg->pending_orphan_variant_dirs, live_dirs);

  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Validate
// ─────────────────────────────────────────────────────────────────────────────

namespace {

void AddFinding(ordered_json* arr, const std::string& code, const std::string& msg) {
  ordered_json e = ordered_json::object();
  e["code"] = code;
  e["message"] = msg;
  arr->push_back(e);
}

}  // namespace

ModelPackageStatus* ModelPackage_Validate(ModelPackage* pkg, int flags,
                                          const char** out_report_json) {
  if (!pkg) return NullArg("pkg");
  if (!out_report_json) return NullArg("out_report_json");
  *out_report_json = nullptr;
  ordered_json report = ordered_json::object();
  report["errors"] = ordered_json::array();
  report["warnings"] = ordered_json::array();
  ordered_json* errors = &report["errors"];
  ordered_json* warnings = &report["warnings"];

  std::error_code ec;

  // SCHEMA: re-validate the in-memory manifest by serializing then re-parsing
  // into a scratch ModelPackage with strict mode. Validates schema for both
  // committed and uncommitted state.
  if (flags & MODEL_PACKAGE_VALIDATE_SCHEMA) {
    // Re-run each component/variant through the parser to confirm shape.
    for (const auto& comp : pkg->components) {
      mp::ComponentRecord scratch;
      auto opts = mp::PathOptionsFor(pkg);
      if (auto* s = mp::ParseComponentBody(pkg->package_root, opts,
                                           /*strict=*/true,
                                           comp->name, comp->body,
                                           comp->component_dir, &scratch)) {
        AddFinding(errors, "SCHEMA", std::string("component '") + comp->name + "': " + ModelPackageStatus_Message(s));
        ModelPackageStatus_Release(s);
      }
    }
  }

  // PATHS: each external component's path on disk; each shared-asset resolved_path exists.
  if (flags & MODEL_PACKAGE_VALIDATE_PATHS) {
    for (const auto& comp : pkg->components) {
      if (comp->storage == mp::ComponentStorage::kExternal) {
        if (!fs::exists(comp->external_path, ec)) {
          AddFinding(warnings, "PATHS",
                     "component '" + comp->name + "' external file does not exist: " +
                         comp->external_path.string());
        }
      }
    }
    for (const auto& rec : pkg->shared_assets) {
      if (!fs::is_directory(rec->resolved_path, ec)) {
        AddFinding(warnings, "PATHS",
                   "shared asset " + rec->uri + " resolved path is not a directory: " +
                       rec->resolved_path.string());
      }
    }
  }

  // ASSET_REHASH: re-hash each on-disk shared asset and compare to its URI.
  if (flags & MODEL_PACKAGE_VALIDATE_ASSET_REHASH) {
    for (const auto& rec : pkg->shared_assets) {
      if (!fs::is_directory(rec->resolved_path, ec)) continue;  // PATHS / REACH covers this.
      std::string computed;
      if (auto* s = mp::ComputeDirectoryAssetUri(rec->resolved_path, &computed)) {
        AddFinding(errors, "ASSET_REHASH",
                   "shared asset " + rec->uri + ": hashing failed: " +
                       ModelPackageStatus_Message(s));
        ModelPackageStatus_Release(s);
        continue;
      }
      if (computed != rec->uri) {
        AddFinding(errors, "ASSET_REHASH",
                   "shared asset " + rec->uri + " on-disk hash differs: " + computed);
      }
    }
  }

  // UNKNOWN_FIELDS: re-run with strict=true (only flags top-level / known scopes).
  if (flags & MODEL_PACKAGE_VALIDATE_UNKNOWN_FIELDS) {
    static const char* kKnown[] = {
        "schema_version", "package_name", "package_version", "description",
        "layout", "components", "shared_assets", "additional_metadata"};
    for (auto it = pkg->manifest.begin(); it != pkg->manifest.end(); ++it) {
      bool found = false;
      for (auto* k : kKnown)
        if (it.key() == k) {
          found = true;
          break;
        }
      if (!found) {
        AddFinding(warnings, "UNKNOWN_FIELDS",
                   "manifest contains unknown field '" + it.key() + "'.");
      }
    }
  }

  pkg->last_validate_report = report.dump(2);
  *out_report_json = pkg->last_validate_report->c_str();
  if (!errors->empty()) {
    return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                      "ModelPackage_Validate: " + std::to_string(errors->size()) +
                          " error(s) found. See out_report_json for details.");
  }
  return nullptr;
}

}  // extern "C"
