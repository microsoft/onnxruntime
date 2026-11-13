// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.List;
import org.apache.maven.artifact.Artifact;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.project.MavenProject;

/** Runs inference from the consumer project's Maven-resolved runtime artifacts. */
public final class MavenResolutionMojo extends AbstractMojo {
  private MavenProject project;
  private String model;
  private String variant;

  @Override
  public void execute() throws MojoExecutionException {
    try {
      List<URL> urls = new ArrayList<>();
      for (Artifact artifact : project.getArtifacts()) {
        File file = artifact.getFile();
        if (file != null && file.isFile()) {
          urls.add(file.toURI().toURL());
        }
      }
      URL pluginJar =
          MavenResolutionMojo.class.getProtectionDomain().getCodeSource().getLocation();
      urls.add(pluginJar);
      try (URLClassLoader consumer =
          new URLClassLoader(
              urls.toArray(new URL[0]), ClassLoader.getSystemClassLoader().getParent())) {
        Class<?> environment = consumer.loadClass("ai.onnxruntime.OrtEnvironment");
        URL origin = environment.getProtectionDomain().getCodeSource().getLocation();
        if (origin.equals(pluginJar)) {
          throw new AssertionError("ONNX Runtime API leaked into the Maven resolver plugin.");
        }
        Class<?> smoke = consumer.loadClass("RuntimeSmokeTest");
        String[] args =
            "cuda".equals(variant) ? new String[] {model, "--cuda"} : new String[] {model};
        smoke.getMethod("main", String[].class).invoke(null, (Object) args);
      }
    } catch (Exception exception) {
      throw new MojoExecutionException("Maven-resolved ONNX Runtime inference failed.", exception);
    }
  }
}
