//
//  ModelPackage.hpp
//  modelpackage
//
//  Copyright © 2021 Apple Inc. All rights reserved.
//

#ifndef ModelPackage_hpp
#define ModelPackage_hpp

#include <string>
#include <memory>
#include <vector>

#if defined(__cplusplus)
extern "C" {
#endif

/** MPL = Model Package Library. */
namespace MPL {

namespace detail {

class ModelPackageItemInfoImpl;
class ModelPackageImpl;

} // namespace detail

class ModelPackageItemInfo {
    
private:
    
    std::shared_ptr<detail::ModelPackageItemInfoImpl> m_modelPackageItemInfoImpl;
        
public:
    
    /** Creates an instance of file info to hold information about a file that exists in a model package. */
    ModelPackageItemInfo(std::shared_ptr<detail::ModelPackageItemInfoImpl> modelPackageItemInfoImpl);
    ~ModelPackageItemInfo();
    
    /** Unique file identifier of the file in the model package. */
    const std::string& identifier() const;
    
    /** Path of the file inside the model package. */
    const std::string& path() const;
    
    /** Name specified while storing the file in the model package. */
    const std::string& name() const;
    
    /** Author specified while storing the file in the model package. */
    const std::string& author() const;
    
    /** Description specified while storing the file in the model package. Defaults to "". */
    const std::string& description() const;
};


class ModelPackage {

private:
    
    std::shared_ptr<detail::ModelPackageImpl> m_modelPackageImpl;
    
public:
    
    /** Creates an instance of model package that exists at the specified path.
        @param path Path of the model package (with extension .mlpackage).
        @param createIfNecessary Create a new model package if one does not exist at the specificed path. Defaults to true.
        @param readOnly The model package will not be mutated Defaults to false.
        @throw Runtime exception if an invalid model package exists at the specified path. */
    explicit ModelPackage(const std::string& path, bool createIfNecessary = true, bool readOnly = false);
    
    ~ModelPackage();
    
    /** Returns the path of the model package. */
    std::string path() const;
    
    /**
     Set a root model in model package. Each model package has a unique root model, which can be retrieved without needing for an identifier.
         @param path Path of the model file.
         @param name Name of the model file.
         @param author Author of the model file. Reverse DNS identifier of the author application is recommended. Example: com.apple.coremltools.
         @param description Optional description to describe the model file.
         @return Unique file identifier that can be used to retrieve the model file.
         @throw a runtime exception if the model package already contains a root model. */
    std::string setRootModel(const std::string& path, const std::string& name, const std::string& author, const std::string& description = "");

    /**
     replace a root model in model package. model package may or may not already contain a root model. Each model package has a unique root model, which can be retrieved without needing for an identifier.
         @param path Path of the model file.
         @param name Name of the model file.
         @param author Author of the model file. Reverse DNS identifier of the author application is recommended. Example: com.apple.coremltools.
         @param description Optional description to describe the model file.
         @return Unique file identifier that can be used to retrieve the model file. */
    std::string replaceRootModel(const std::string& path, const std::string& name, const std::string& author, const std::string& description = "");

    /**
     Retrieve previously set root model from the model package.
         @return ModelPackageItemInfo with information about the retrieved root model file.
         @throw Runtime exception if the model package does not contain a root model. */
    std::shared_ptr<ModelPackageItemInfo> getRootModel() const;
    
    /**
     Add a file or directory in the model package using name and author as a uniqueing key.
         @param path Path of the file.
         @param name Name of the file.
         @param author Author of the file. Reverse DNS identifier of the author application is recommended. Example: com.apple.coremltools.
         @param description Optional description to describe the file.
         @return Unique file identifier that can be used to look up the file.
         @throw a runtime exception if the model package already contains a file with provided name and author. */
    std::string addItem(const std::string& path, const std::string& name, const std::string& author, const std::string& description = "");
    
    /**
     Retrieve previously added file or directory from the model package by providing an identifier.
        @param identifier Unique identifier of a previous added file
        @return A pointer to ModelPackageItemInfo with information about the retrieved file or directory. nullptr if a file or directory with given identifier does not exist. */
    std::shared_ptr<ModelPackageItemInfo> findItem(const std::string& identifier) const;
    
    /**
     Retrieve previously added file or directory from the model package by providing name and author.
        @param name Name of a previous added file
        @param author Author of a previous added file
        @return A pointer to ModelPackageItemInfo with information about the retrieved file or directory by providing name and author. nullptr if a file or directory with given name and author does not exist. */
    std::shared_ptr<ModelPackageItemInfo> findItem(const std::string& name, const std::string& author) const;
    
    /**
     Retrieve previously added files or directories from the model package by providing an author.
        @param author Name of the author.
        @return Vector of ModelPackageItemInfo objects with information about the retrieved files by providing the author. */
    std::vector<ModelPackageItemInfo> findItemsByAuthor(const std::string& author) const;
    
    /**
     Remove previously added file or directory from the model package by providing an identifier.
        @param identifier Unique file identifier corresponding to a file that was added previously.
        @throw Runtime exception if the model package does not contain file with provided identifier. */
    void removeItem(const std::string& identifier);
    
    /**
     Tells if the input path corresponds to a valid model package.
        @param path Path of model package.
        @return True if the path corresponds to a valid model package. False, otherwise. */
    static bool isValid(const std::string& path);
    
    /**
     Creates an empty file in the model package and returns corresponding file identifier.
        @param name Name of the file.
        @param author Author of the file. Reverse DNS identifier of the author application is recommended. Example: com.apple.coremltools.
        @param description Optional description to describe the file.
        @return ModelPackageItemInfo with information about the created file. */
    ModelPackageItemInfo createFile(const std::string& name, const std::string& author, const std::string& description);
};

} // namespace MPL
    
#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* ModelPackage_hpp */

