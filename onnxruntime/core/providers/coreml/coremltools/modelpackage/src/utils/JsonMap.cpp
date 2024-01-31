//  JsonMap.cpp
//  modelpackage
//
//  Copyright © 2021 Apple. All rights reserved.

#include <iostream>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "JsonMap.hpp"
#include "json.hpp"

using namespace nlohmann;

class JsonMapImpl {
    
public:
    
    nlohmann::json m_jsonObject;
    
    JsonMapImpl();
    JsonMapImpl(std::istream& stream);
    JsonMapImpl(nlohmann::json j_init);
    
    JsonMapImpl(const JsonMapImpl&) = delete;
    JsonMapImpl(JsonMapImpl&&) = delete;
    JsonMapImpl& operator=(const JsonMapImpl&) = delete;
    JsonMapImpl& operator=(JsonMapImpl&&) = delete;
    
    /* ==== Key operations ==== */

    bool hasKey(const std::string& key) const;
    void getKeys(std::vector<std::string>& keys);

    /* ==== Getter methods ==== */

    std::string getString(const std::string& key) const;
    std::unique_ptr<JsonMapImpl> getObject(const std::string& key) const;

    /* ==== Setter methods ==== */

    void setString(const std::string& key, const std::string& value);
    void setObject(const std::string& key, std::unique_ptr<JsonMapImpl> value);
    
    void serialize(std::ostream& stream);
    void deserialize(std::istream& stream);
};

JsonMapImpl::JsonMapImpl() {
    m_jsonObject = nlohmann::json({});
}

JsonMapImpl::JsonMapImpl(std::istream& stream) {
    deserialize(stream);
}

JsonMapImpl::JsonMapImpl(nlohmann::json j_init)
: m_jsonObject(j_init) {
}

/* ==== Key operations ==== */

bool JsonMapImpl::hasKey(const std::string& key) const {
    return m_jsonObject.count(key) > 0;
}

void JsonMapImpl::getKeys(std::vector<std::string>& keys) {
    for(json::iterator it = m_jsonObject.begin(); it != m_jsonObject.end(); ++it) {
        keys.push_back(it.key());
    }
}

/* ==== Getter methods ==== */

std::string JsonMapImpl::getString(const std::string& key) const {
    return m_jsonObject.at(key).get<std::string>();
}

std::unique_ptr<JsonMapImpl> JsonMapImpl::getObject(const std::string& key) const {
    auto childCopy = m_jsonObject.at(key);
    return std::make_unique<JsonMapImpl>(childCopy);
}

/* ==== Setter methods ==== */

void JsonMapImpl::setString(const std::string& key, const std::string& value) {
    m_jsonObject[key] = value;
}

void JsonMapImpl::setObject(const std::string& key, std::unique_ptr<JsonMapImpl> value) {
    m_jsonObject[key] = value->m_jsonObject;
}

void JsonMapImpl::deserialize(std::istream& stream) {
    if(!stream.good()) {
        throw std::runtime_error("Input stream is not valid");
    }

    try {
        stream >> m_jsonObject;
    } catch (std::exception& e) {
        // nlohmann::json raises std::exception on parser errors, but the client of JsonMap only
        // handles std::runtime_error because they don't want to "handle" programming errors
        // (std::logic_error).
        //
        // As such, we translate the exception type here.
        throw std::runtime_error(e.what());
    }
}

void JsonMapImpl::serialize(std::ostream& stream) {
    // write prettified JSON to another file
    stream << std::setw(4) << m_jsonObject << std::endl;

}

/* ==== JsonMap ==== */

JsonMap::JsonMap()
: m_jsonMapImpl(std::make_unique<JsonMapImpl>())
{
}

JsonMap::JsonMap(std::istream& stream)
: m_jsonMapImpl(std::make_unique<JsonMapImpl>(stream))
{
}

JsonMap::JsonMap(std::unique_ptr<JsonMapImpl> jsonMapImpl)
: m_jsonMapImpl(std::move(jsonMapImpl))
{
}

JsonMap::~JsonMap() = default;

/* ==== Key operations ==== */

bool JsonMap::hasKey(const std::string& key) const {
    return m_jsonMapImpl->hasKey(key);
}

void JsonMap::getKeys(std::vector<std::string>& keys) {
    return m_jsonMapImpl->getKeys(keys);
}

/* ==== Getter methods ==== */

std::string JsonMap::getString(const std::string& key) const {
    return m_jsonMapImpl->getString(key);
}

std::unique_ptr<JsonMap> JsonMap::getObject(const std::string& key) const {
    return std::make_unique<JsonMap>(m_jsonMapImpl->getObject(key));
}

/* ==== Setter methods ==== */

void JsonMap::setString(const std::string& key, const std::string& value) {
    return m_jsonMapImpl->setString(key, value);
}

void JsonMap::setObject(const std::string& key, std::unique_ptr<JsonMap> value) {
    m_jsonMapImpl->setObject(key, std::move(value->m_jsonMapImpl));
}

void JsonMap::serialize(std::ostream& stream) {
    return m_jsonMapImpl->serialize(stream);
}
