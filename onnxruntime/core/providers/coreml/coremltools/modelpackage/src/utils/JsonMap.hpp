//
//  JsonMap.hpp
//  modelpackage
//
//  Copyright © 2021 Apple. All rights reserved.
//

#pragma once

#include <iostream>
#include <vector>
#include <string>

class JsonMapImpl;

class JsonMap {
    
private:
    
    std::unique_ptr<JsonMapImpl> m_jsonMapImpl;
    
public:

    JsonMap();
    JsonMap(std::istream& stream);
    JsonMap(std::unique_ptr<JsonMapImpl> jsonMapImpl);
    
    ~JsonMap();
    
    JsonMap(const JsonMap&) = delete;
    JsonMap(JsonMap&&) = delete;
    JsonMap& operator=(const JsonMap&) = delete;
    JsonMap& operator=(JsonMap&&) = delete;

    /* ==== Key operations ==== */

    bool hasKey(const std::string& key) const;
    void getKeys(std::vector<std::string>& keys);

    /* ==== Getter methods ==== */

    std::string getString(const std::string& key) const;
    std::unique_ptr<JsonMap> getObject(const std::string& key) const;

    /* ==== Setter methods ==== */

    void setString(const std::string& key, const std::string& value);
    void setObject(const std::string& key, std::unique_ptr<JsonMap> value);

    void serialize(std::ostream& stream);
};
