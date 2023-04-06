# Objective-C API Documentation

The API should be documented with comments in the [public header files](../include).

## Documentation Generation

The [Jazzy](https://github.com/realm/jazzy) tool is used to generate documentation from the code.

To generate documentation, from the repo root, run:

```bash
jazzy --config objectivec/docs/jazzy_config.yaml --output <output directory>
```

The generated documentation website files will be in `<output directory>`.

[main_page.md](./main_page.md) contains content for the main page of the generated documentation website.
