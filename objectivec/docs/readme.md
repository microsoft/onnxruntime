# Objective-C API Documentation

The API should be documented with comments in the [public header files](../include).

## Documentation Generation

The [Jazzy](https://github.com/realm/jazzy) tool is used to generate documentation from the code.

For example, to generate documentation for a release version, from the repo root, run:

```bash
jazzy --config objectivec/docs/jazzy_config.yaml --output <output directory> --module-version $(cat VERSION_NUMBER)
```

The generated documentation website files will be in `<output directory>`.

[docs.readme.md](./docs.readme.md) contains content for the main page of the generated documentation website.
