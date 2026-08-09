/*
 * Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Contains the metadata associated with an ONNX model.
 *
 * <p>Unspecified default fields contain the empty string.
 *
 * <p>This class is a Java side copy of the native metadata, it does not access the native runtime.
 */
public final class OnnxModelMetadata {

  private final String producerName;
  private final String graphName;
  private final String graphDescription;
  private final String domain;
  private final String description;
  private final long version;

  private final Map<String, String> customMetadata;

  /**
   * Constructed by an OrtSession in native code, nulls are replaced with the empty string or empty
   * map as appropriate.
   *
   * @param producerName The model producer name.
   * @param graphName The model graph name.
   * @param graphDescription The model graph description.
   * @param domain The model domain name.
   * @param description The model description.
   * @param version The model version.
   * @param customMetadataArray Any custom metadata associated with the model.
   */
  OnnxModelMetadata(
      String producerName,
      String graphName,
      String graphDescription,
      String domain,
      String description,
      long version,
      String[] customMetadataArray) {
    this.producerName = producerName == null ? "" : producerName;
    this.graphName = graphName == null ? "" : graphName;
    this.graphDescription = graphDescription == null ? "" : graphDescription;
    this.domain = domain == null ? "" : domain;
    this.description = description == null ? "" : description;
    this.version = version;
    if (customMetadataArray != null && customMetadataArray.length > 0) {
      this.customMetadata = new HashMap<>();
      if (customMetadataArray.length % 2 == 1) {
        throw new IllegalStateException(
            "Asked for keys and values, but received an odd number of elements.");
      }
      for (int i = 0; i < customMetadataArray.length; i += 2) {
        customMetadata.put(customMetadataArray[i], customMetadataArray[i + 1]);
      }
    } else {
      this.customMetadata = Collections.emptyMap();
    }
  }

  /**
   * Constructed by an OrtSession, nulls are replaced with the empty string or empty map as
   * appropriate.
   *
   * @param producerName The model producer name.
   * @param graphName The model graph name.
   * @param graphDescription The model graph name.
   * @param domain The model domain name.
   * @param description The model description.
   * @param version The model version.
   * @param customMetadata Any custom metadata associated with the model.
   */
  OnnxModelMetadata(
      String producerName,
      String graphName,
      String graphDescription,
      String domain,
      String description,
      long version,
      Map<String, String> customMetadata) {
    this.producerName = producerName == null ? "" : producerName;
    this.graphName = graphName == null ? "" : graphName;
    this.graphDescription = graphDescription == null ? "" : graphDescription;
    this.domain = domain == null ? "" : domain;
    this.description = description == null ? "" : description;
    this.version = version;
    this.customMetadata = customMetadata == null ? Collections.emptyMap() : customMetadata;
  }

  /**
   * Copy constructor.
   *
   * @param other The metadata to copy.
   */
  public OnnxModelMetadata(OnnxModelMetadata other) {
    this.producerName = other.producerName;
    this.graphName = other.graphName;
    this.graphDescription = other.graphDescription;
    this.domain = other.domain;
    this.description = other.description;
    this.version = other.version;
    this.customMetadata =
        other.customMetadata.isEmpty()
            ? Collections.emptyMap()
            : new HashMap<>(getCustomMetadata());
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    OnnxModelMetadata that = (OnnxModelMetadata) o;
    return version == that.version
        && producerName.equals(that.producerName)
        && graphName.equals(that.graphName)
        && graphDescription.equals(that.graphDescription)
        && domain.equals(that.domain)
        && description.equals(that.description)
        && customMetadata.equals(that.customMetadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        producerName, graphName, graphDescription, domain, description, version, customMetadata);
  }

  /**
   * Gets the producer name.
   *
   * @return The producer name.
   */
  public String getProducerName() {
    return producerName;
  }

  /**
   * Gets the graph name.
   *
   * @return The graph name.
   */
  public String getGraphName() {
    return graphName;
  }

  /**
   * Gets the graph description.
   *
   * @return The graph description.
   */
  public String getGraphDescription() {
    return graphDescription;
  }

  /**
   * Gets the domain.
   *
   * @return The domain.
   */
  public String getDomain() {
    return domain;
  }

  /**
   * Gets the model description.
   *
   * @return The description.
   */
  public String getDescription() {
    return description;
  }

  /**
   * Gets the model version.
   *
   * @return The model version.
   */
  public long getVersion() {
    return version;
  }

  /**
   * Gets an unmodifiable reference to the complete custom metadata.
   *
   * @return The custom metadata.
   */
  public Map<String, String> getCustomMetadata() {
    return Collections.unmodifiableMap(customMetadata);
  }

  /**
   * Returns Optional.of(value) if the custom metadata has a value for the supplied key, otherwise
   * returns {@link Optional#empty}.
   *
   * @param key The custom metadata key.
   * @return The custom metadata value if present.
   */
  public Optional<String> getCustomMetadataValue(String key) {
    return Optional.ofNullable(customMetadata.get(key));
  }

  @Override
  public String toString() {
    return "OnnxModelMetadata{"
        + "producerName='"
        + producerName
        + '\''
        + ", graphName='"
        + graphName
        + '\''
        + ", graphDescription='"
        + graphDescription
        + '\''
        + ", domain='"
        + domain
        + '\''
        + ", description='"
        + description
        + '\''
        + ", version="
        + version
        + ", customMetadata="
        + customMetadata
        + '}';
  }
}
