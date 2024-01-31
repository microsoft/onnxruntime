//
//  ModelPackage.cpp
//  modelpackage
//
//  Copyright © 2021 Apple Inc. All rights reserved.
//

#include "ModelPackage.hpp"

#include "utils/JsonMap.hpp"

#include <algorithm>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <istream>
#include <string>

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#else
#error "missing required header <filesystem>"
#endif

// ORT_EDIT: Use UuidCreate on Windows.
#if defined(_WIN32)
#pragma comment(lib, "rpcrt4.lib")  // UuidCreate
#include <windows.h>
#else
#include <uuid/uuid.h>
#endif
#include <vector>

#if defined(__cplusplus)
extern "C" {
#endif

static const char* kModelPackageManifestFileName = "Manifest.json";
static const char* kModelPackageFileFormatVersionKey = "fileFormatVersion";

static const int kModelPackageFileFormatMajorVersion = 1;
static const int kModelPackageFileFormatMinorVersion = 0;
static const int kModelPackageFileFormatPatchVersion = 0;

static const char* kModelPackageItemInfoEntriesKey = "itemInfoEntries";

static const char* kModelPackageItemInfoPathKey = "path";
static const char* kModelPackageItemInfoNameKey = "name";
static const char* kModelPackageItemInfoAuthorKey = "author";
static const char* kModelPackageItemInfoDescriptionKey = "description";

static const char* kModelPackageDataDir = "Data";

static const char* kModelPackageRootModelKey = "rootModelIdentifier";

using namespace MPL;
using namespace detail;
using namespace std::filesystem;

class detail::ModelPackageItemInfoImpl {
 private:
  std::string m_identifier;
  std::string m_path;
  std::string m_name;
  std::string m_author;
  std::string m_description;

 public:
  ModelPackageItemInfoImpl(const std::string& identifier, const std::string& path, const std::string& name, const std::string& author, const std::string& description);

  ~ModelPackageItemInfoImpl();

  inline const std::string& identifier() {
    return m_identifier;
  }

  inline const std::string& path() {
    return m_path;
  }

  inline const std::string& name() {
    return m_name;
  }

  inline const std::string& author() {
    return m_author;
  }

  inline const std::string& description() {
    return m_description;
  }
};

ModelPackageItemInfoImpl::ModelPackageItemInfoImpl(const std::string& identifier, const std::string& path, const std::string& name, const std::string& author, const std::string& description)
    : m_identifier(identifier),
      m_path(path),
      m_name(name),
      m_author(author),
      m_description(description) {
}

ModelPackageItemInfoImpl::~ModelPackageItemInfoImpl() {
}

ModelPackageItemInfo::ModelPackageItemInfo(std::shared_ptr<ModelPackageItemInfoImpl> modelPackageItemInfoImpl)
    : m_modelPackageItemInfoImpl(modelPackageItemInfoImpl) {
}

ModelPackageItemInfo::~ModelPackageItemInfo() {
}

const std::string& ModelPackageItemInfo::identifier() const {
  return m_modelPackageItemInfoImpl->identifier();
}

const std::string& ModelPackageItemInfo::path() const {
  return m_modelPackageItemInfoImpl->path();
}

const std::string& ModelPackageItemInfo::name() const {
  return m_modelPackageItemInfoImpl->name();
}

const std::string& ModelPackageItemInfo::author() const {
  return m_modelPackageItemInfoImpl->author();
}

const std::string& ModelPackageItemInfo::description() const {
  return m_modelPackageItemInfoImpl->description();
}

class detail::ModelPackageImpl {
 private:
  std::filesystem::path m_packagePath;
  std::filesystem::path m_manifestPath;
  std::filesystem::path m_packageDataDirPath;

  std::unique_ptr<JsonMap> m_manifest;

  bool m_readOnly;

  void validate();

  std::unique_ptr<JsonMap> getItemInfoEntries() const;
  std::unique_ptr<JsonMap> getItemInfoEntry(const std::string& identifier) const;

  void createItemInfoEntry(const std::string& identifier, const std::string& path, const std::string& name, const std::string& author, const std::string& description);
  void removeItemInfoEntry(const std::string& identifier);

  std::string generateIdentifier() const;

  std::filesystem::path getItemPath(const std::string& name, const std::string& author) const;

 public:
  ModelPackageImpl(const std::filesystem::path& path, bool createIfNecessary = true, bool readOnly = false);
  ~ModelPackageImpl();

  inline const std::filesystem::path& path() const {
    return m_packagePath;
  }

  std::string setRootModel(const std::filesystem::path& path, const std::string& name, const std::string& author, const std::string& description);
  std::string replaceRootModel(const std::filesystem::path& path, const std::string& name, const std::string& author, const std::string& description);
  std::shared_ptr<ModelPackageItemInfo> getRootModel() const;

  std::string addItem(const std::filesystem::path& path, const std::string& name, const std::string& author, const std::string& description);
  std::shared_ptr<ModelPackageItemInfo> findItem(const std::string& identifier) const;
  std::shared_ptr<ModelPackageItemInfo> findItem(const std::string& name, const std::string& author) const;
  std::vector<ModelPackageItemInfo> findItemsByAuthor(const std::string& author) const;

  void removeItem(const std::string& identifier);
  static bool isValid(const std::filesystem::path& path);

  ModelPackageItemInfo createFile(const std::string& name, const std::string& author, const std::string& description);
};

#if defined(__APPLE__)
#pragma mark ModelPackageImpl
#endif

ModelPackageImpl::ModelPackageImpl(const std::filesystem::path& path, bool createIfNecessary, bool readOnly)
    : m_packagePath(path),
      m_manifestPath(path / kModelPackageManifestFileName),
      m_packageDataDirPath(path / kModelPackageDataDir),
      m_manifest(nullptr),
      m_readOnly(readOnly) {
  if (std::filesystem::exists(m_packagePath)) {
    if (std::filesystem::exists(m_manifestPath)) {
      std::ifstream manifestStream(m_manifestPath, std::ios::binary);
      m_manifest = std::make_unique<JsonMap>(manifestStream);
      manifestStream.close();
    } else {
      throw std::runtime_error("A valid manifest does not exist at path: " + m_manifestPath.string());
    }
  }
  // Create the package structure at specified path
  else if (createIfNecessary) {
    if (false == create_directory(m_packagePath)) {
      throw std::runtime_error("Failed to create model package at path: " + m_packagePath.string());
    }

    if (false == create_directory(m_packageDataDirPath)) {
      throw std::runtime_error("Failed to create data directory at path: " + m_packageDataDirPath.string());
    }

    m_manifest = std::make_unique<JsonMap>();
    std::stringstream ss;
    ss << kModelPackageFileFormatMajorVersion << "." << kModelPackageFileFormatMinorVersion << "." << kModelPackageFileFormatPatchVersion;
    m_manifest->setString(kModelPackageFileFormatVersionKey, ss.str());
  }
  // Error out since package does not exist
  else {
    throw std::runtime_error("Failed to open model package at path: " + m_packagePath.string());
  }

  validate();
}

ModelPackageImpl::~ModelPackageImpl() {
  if (m_readOnly) {
    return;
  }

  std::filesystem::path uniquedDestination(m_manifestPath);
  std::filesystem::path suffix(generateIdentifier());  // std::filesystem::path from stringified UUID
  uniquedDestination.replace_extension(suffix);        // unique filename in the presumed writable directory where Manifest.json is sited

  std::ofstream uniquedStream(uniquedDestination, std::ios::binary);
  m_manifest->serialize(uniquedStream);
  uniquedStream.close();
  if (uniquedStream.fail()) {  // If any of the above fail do not go on to move uniquedDestination to m_manifestPath.
    return;
  }

  std::error_code ecode;
  std::filesystem::rename(uniquedDestination, m_manifestPath, ecode);  // On failure sets ecode and makes no changes. Does not throw.
  if (ecode.value()) {
    std::filesystem::remove(uniquedDestination);
  }
}

void ModelPackageImpl::validate() {
  const std::string versionString = m_manifest->getString(kModelPackageFileFormatVersionKey);

  std::istringstream versionStringStream(versionString);
  std::vector<std::string> versionTokens;
  for (std::string token; std::getline(versionStringStream, token, '.');) {
    versionTokens.push_back(token);
  }

  if (versionTokens.size() != 3) {
    throw std::runtime_error("File format version must be in the form of major.minor.patch, but the specified value was: " + versionString);
  }

  int majorVersion = 0;
  int minorVersion = 0;
  int patchVersion = 0;
  try {
    majorVersion = std::stoi(versionTokens[0]);
    minorVersion = std::stoi(versionTokens[1]);
    patchVersion = std::stoi(versionTokens[2]);
  } catch (std::invalid_argument& e) {
    throw std::runtime_error("Failed to parse file format version: " + versionString + " because: " + e.what());
  }

  if (majorVersion < 0 ||
      minorVersion < 0 ||
      patchVersion < 0) {
    throw std::runtime_error("File format version uses negative number(s): " + versionString);
  }

  if ((majorVersion > kModelPackageFileFormatMajorVersion) ||
      (majorVersion == kModelPackageFileFormatMajorVersion && minorVersion > kModelPackageFileFormatMinorVersion) ||
      (minorVersion == kModelPackageFileFormatMinorVersion && patchVersion > kModelPackageFileFormatPatchVersion)) {
    throw std::runtime_error("Unsupported version: " + versionString);
  }

  // Validate 1.0.0 model package

  auto itemInfoEntries = getItemInfoEntries();
  if (itemInfoEntries != nullptr) {
    std::vector<std::string> identifiers;
    itemInfoEntries->getKeys(identifiers);
    for (const auto& identifier : identifiers) {
      auto itemInfoEntry = getItemInfoEntry(identifier);

      if (false == itemInfoEntry->hasKey(kModelPackageItemInfoPathKey) ||
          false == itemInfoEntry->hasKey(kModelPackageItemInfoNameKey) ||
          false == itemInfoEntry->hasKey(kModelPackageItemInfoAuthorKey) ||
          false == itemInfoEntry->hasKey(kModelPackageItemInfoDescriptionKey)) {
        throw std::runtime_error("Invalid itemInfo for identifier: " + identifier);
      }

      auto path = m_packageDataDirPath / itemInfoEntry->getString(kModelPackageItemInfoPathKey);
      if (false == exists(path)) {
        throw std::runtime_error("Item does not exist for identifier: " + identifier);
      }
    }
  }
}

std::unique_ptr<JsonMap> ModelPackageImpl::getItemInfoEntries() const {
  if (m_manifest->hasKey(kModelPackageItemInfoEntriesKey)) {
    return m_manifest->getObject(kModelPackageItemInfoEntriesKey);
  }

  return std::make_unique<JsonMap>();
}

std::unique_ptr<JsonMap> ModelPackageImpl::getItemInfoEntry(const std::string& identifier) const {
  auto itemInfoEntries = getItemInfoEntries();

  if (itemInfoEntries->hasKey(identifier)) {
    return itemInfoEntries->getObject(identifier);
  }

  return nullptr;
}

void ModelPackageImpl::removeItemInfoEntry(const std::string& identifier) {
  auto itemInfoEntries = getItemInfoEntries();

  std::vector<std::string> identifiers;
  itemInfoEntries->getKeys(identifiers);

  auto newItemInfoEntries = std::make_unique<JsonMap>();
  for (const auto& localIdentifier : identifiers) {
    if (localIdentifier != identifier) {
      newItemInfoEntries->setObject(localIdentifier, itemInfoEntries->getObject(localIdentifier));
    }
  }

  m_manifest->setObject(kModelPackageItemInfoEntriesKey, std::move(newItemInfoEntries));
}

void ModelPackageImpl::createItemInfoEntry(const std::string& identifier, const std::string& path, const std::string& name, const std::string& author, const std::string& description) {
  auto itemInfoEntry = getItemInfoEntry(identifier);

  if (nullptr == itemInfoEntry) {
    itemInfoEntry = std::make_unique<JsonMap>();
  }

  itemInfoEntry->setString(kModelPackageItemInfoPathKey, path);
  itemInfoEntry->setString(kModelPackageItemInfoNameKey, name);
  itemInfoEntry->setString(kModelPackageItemInfoAuthorKey, author);
  itemInfoEntry->setString(kModelPackageItemInfoDescriptionKey, description);

  auto itemInfoEntries = getItemInfoEntries();
  itemInfoEntries->setObject(identifier, std::move(itemInfoEntry));
  m_manifest->setObject(kModelPackageItemInfoEntriesKey, std::move(itemInfoEntries));
}

std::filesystem::path ModelPackageImpl::getItemPath(const std::string& name, const std::string& author) const {
  return std::filesystem::path(author) / name;
}

std::string ModelPackageImpl::generateIdentifier() const {
#if defined(_WIN32)
  UUID uuid;
  UuidCreate(&uuid);

  RPC_CSTR uuidStr;
  UuidToStringA(&uuid, &uuidStr);

  std::string uuidStrCpp(reinterpret_cast<char*>(uuidStr));

  RpcStringFreeA(&uuidStr);

  return uuidStrCpp;
#else
  uuid_t uuid;

  // uuid_unparse generates a 36-character null-terminated string (37 bytes).
  // they provide no mechanisms for us to deduce this length, therefore
  // we have to hardcode it here.
  char buf[37] = "";

  uuid_generate(uuid);
  uuid_unparse(uuid, buf);

  return std::string(buf);
#endif
}

ModelPackageItemInfo ModelPackageImpl::createFile(const std::string& name, const std::string& author, const std::string& description) {
  if (findItem(name, author) != nullptr) {
    throw std::runtime_error("The package already contains a file with name: " + name + " author: " + author);
  }

  auto filePath = getItemPath(name, author);
  auto dstPath = m_packageDataDirPath / filePath;

  create_directories(dstPath.parent_path());

  std::ofstream stream(dstPath, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("Failed to create file at path: " + dstPath.string());
  }

  auto identifier = generateIdentifier();
  createItemInfoEntry(identifier, filePath.string(), name, author, description);
  return *(findItem(identifier));
}

std::string ModelPackageImpl::addItem(const std::filesystem::path& path, const std::string& name, const std::string& author, const std::string& description) {
  if (findItem(name, author) != nullptr) {
    throw std::runtime_error("The package already contains a file with name: " + name + " author: " + author);
  }

  auto filePath = getItemPath(name, author);
  auto dstPath = m_packageDataDirPath / filePath;

  create_directories(dstPath.parent_path());
  std::filesystem::copy(path, dstPath);

  auto identifier = generateIdentifier();
  createItemInfoEntry(identifier, filePath.string(), name, author, description);
  return identifier;
}

std::string ModelPackageImpl::setRootModel(const std::filesystem::path& path, const std::string& name, const std::string& author, const std::string& description) {
  if (m_manifest->hasKey(kModelPackageRootModelKey)) {
    throw std::runtime_error("A root model already exists in this package");
  }

  auto identifier = addItem(path, name, author, description);
  m_manifest->setString(kModelPackageRootModelKey, identifier);
  return identifier;
}

std::string ModelPackageImpl::replaceRootModel(const std::filesystem::path& path, const std::string& name, const std::string& author, const std::string& description) {
  if (m_manifest->hasKey(kModelPackageRootModelKey)) {
    auto rootModelIdentifier = m_manifest->getString(kModelPackageRootModelKey);
    removeItem(rootModelIdentifier);
  }

  auto identifier = addItem(path, name, author, description);
  m_manifest->setString(kModelPackageRootModelKey, identifier);
  return identifier;
}

std::shared_ptr<ModelPackageItemInfo> ModelPackageImpl::getRootModel() const {
  if (false == m_manifest->hasKey(kModelPackageRootModelKey)) {
    throw std::runtime_error("Failed to look up root model");
  }

  auto rootModelIdentifier = m_manifest->getString(kModelPackageRootModelKey);
  return findItem(rootModelIdentifier);
}

std::shared_ptr<ModelPackageItemInfo> ModelPackageImpl::findItem(const std::string& identifier) const {
  auto itemInfoEntry = getItemInfoEntry(identifier);
  if (itemInfoEntry == nullptr) {
    return nullptr;
  }

  auto path = m_packageDataDirPath / itemInfoEntry->getString(kModelPackageItemInfoPathKey);
  auto name = itemInfoEntry->getString(kModelPackageItemInfoNameKey);
  auto author = itemInfoEntry->getString(kModelPackageItemInfoAuthorKey);
  auto description = itemInfoEntry->getString(kModelPackageItemInfoDescriptionKey);

// ORT_EDIT: need to use path.string() on Windows
#if defined(_WIN32)
  return std::make_shared<ModelPackageItemInfo>(std::make_shared<ModelPackageItemInfoImpl>(identifier, path.string(), name, author, description));

#else
  return std::make_shared<ModelPackageItemInfo>(std::make_shared<ModelPackageItemInfoImpl>(identifier, path, name, author, description));
#endif
}

std::shared_ptr<ModelPackageItemInfo> ModelPackageImpl::findItem(const std::string& name, const std::string& author) const {
  auto itemInfoEntries = getItemInfoEntries();
  if (itemInfoEntries != nullptr) {
    std::vector<std::string> identifiers;
    itemInfoEntries->getKeys(identifiers);
    for (const auto& identifier : identifiers) {
      auto itemInfo = findItem(identifier);
      if (itemInfo->author() == author && itemInfo->name() == name) {
        return itemInfo;
      }
    }
  }

  return nullptr;
}

std::vector<ModelPackageItemInfo> ModelPackageImpl::findItemsByAuthor(const std::string& author) const {
  auto itemInfoVector = std::vector<ModelPackageItemInfo>();
  auto itemInfoEntries = getItemInfoEntries();
  if (itemInfoEntries != nullptr) {
    std::vector<std::string> identifiers;
    itemInfoEntries->getKeys(identifiers);
    for (const auto& identifier : identifiers) {
      auto itemInfo = findItem(identifier);
      if (itemInfo->author() == author) {
        itemInfoVector.push_back(*itemInfo);
      }
    }
  }

  return itemInfoVector;
}

void ModelPackageImpl::removeItem(const std::string& identifier) {
  auto itemInfoEntry = getItemInfoEntry(identifier);
  if (itemInfoEntry == nullptr) {
    throw std::runtime_error("Failed to look up file with identifier: " + identifier);
  }

  auto path = m_packageDataDirPath / itemInfoEntry->getString(kModelPackageItemInfoPathKey);
  // ORT_EDIT: std::remove doesn't work on Windows. Use std::filesystem::remove instead.
  // if (0 != std::remove(path.c_str())) {
  if (!std::filesystem::remove(path)) {
    throw std::runtime_error("Failed to remove file at path: " + path.string());
  }

  removeItemInfoEntry(identifier);
}

bool ModelPackageImpl::isValid(const std::filesystem::path& path) {
  try {
    ModelPackageImpl(path, false, true);
  } catch (std::runtime_error& /*e*/) {  // ORT_EDIT: comment out unused variable
    return false;
  }
  return true;
}

// ORT_EDIT: pragma only available on APPLE platforms
#if defined(__APPLE__)
#pragma mark ModelPackage
#endif

ModelPackage::ModelPackage(const std::string& packagePath, bool createIfNecessary, bool readOnly)
    : m_modelPackageImpl(std::make_shared<ModelPackageImpl>(packagePath, createIfNecessary, readOnly)) {
}

ModelPackage::~ModelPackage() {
}

std::string ModelPackage::path() const {
// ORT_EDIT: Windows doesn't automatically convert to std::string as the native format could be char or wchar.
#if defined(_WIN32)
  return m_modelPackageImpl->path().string();
#else
  return m_modelPackageImpl->path();
#endif
}

std::string ModelPackage::setRootModel(const std::string& path, const std::string& name, const std::string& author, const std::string& description) {
  return m_modelPackageImpl->setRootModel(path, name, author, description);
}

std::string ModelPackage::replaceRootModel(const std::string& path, const std::string& name, const std::string& author, const std::string& description) {
  return m_modelPackageImpl->replaceRootModel(path, name, author, description);
}

std::shared_ptr<ModelPackageItemInfo> ModelPackage::getRootModel() const {
  return m_modelPackageImpl->getRootModel();
}

std::string ModelPackage::addItem(const std::string& path, const std::string& name, const std::string& author, const std::string& description) {
  return m_modelPackageImpl->addItem(path, name, author, description);
}

std::shared_ptr<ModelPackageItemInfo> ModelPackage::findItem(const std::string& identifier) const {
  return m_modelPackageImpl->findItem(identifier);
}

std::shared_ptr<ModelPackageItemInfo> ModelPackage::findItem(const std::string& name, const std::string& author) const {
  return m_modelPackageImpl->findItem(name, author);
}

std::vector<ModelPackageItemInfo> ModelPackage::findItemsByAuthor(const std::string& author) const {
  return m_modelPackageImpl->findItemsByAuthor(author);
}

void ModelPackage::removeItem(const std::string& identifier) {
  return m_modelPackageImpl->removeItem(identifier);
}

bool ModelPackage::isValid(const std::string& path) {
  return ModelPackageImpl::isValid(path);
}

ModelPackageItemInfo ModelPackage::createFile(const std::string& name, const std::string& author, const std::string& description) {
  return m_modelPackageImpl->createFile(name, author, description);
}

#if defined(__cplusplus)
}  // extern "C"
#endif
