# Objective-C API Documentation

The API should be documented with comments in the [public header files](../include).

## Documentation Generation

Documentation generation dependencies are defined in [Gemfile](./Gemfile) and managed with
[Bundler](https://bundler.io/).

The [Jazzy](https://github.com/realm/jazzy) tool is used to generate documentation from the code.

To install the dependencies, from the repo root, run:

```bash
BUNDLE_GEMFILE=objectivec/docs/Gemfile bundle install
```

Then, to generate the documentation, run:

```bash
BUNDLE_GEMFILE=objectivec/docs/Gemfile bundle exec jazzy \
	--config objectivec/docs/jazzy_config.yaml \
	--output <output directory>
```

The generated documentation website files will be in `<output directory>`.

[main_page.md](./main_page.md) contains content for the main page of the generated documentation website.
