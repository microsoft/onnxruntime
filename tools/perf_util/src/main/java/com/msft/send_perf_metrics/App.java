package com.msft.send_perf_metrics;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.*;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

import java.sql.Connection;
import java.sql.Types;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class App {

	static String exec_command(Path source_dir, String... commands) throws Exception {
		ProcessBuilder sb = new ProcessBuilder(commands).directory(source_dir.toFile()).redirectErrorStream(true);
		Process p = sb.start();
		if (p.waitFor() != 0)
			throw new RuntimeException("execute " + String.join(" ", commands) + " failed");
		try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
			return r.readLine();
		}
	}

	public static void main(String[] args) throws Exception {

		final Path source_dir = Paths.get(args[0]);
		final List<Path> perf_metrics = new ArrayList<Path>();
		Files.walkFileTree(source_dir, new SimpleFileVisitor<Path>() {

			@Override
			public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
				String dirname = dir.getFileName().toString();
				if (dirname != "." && dirname.startsWith("."))
					return FileVisitResult.SKIP_SUBTREE;
				return FileVisitResult.CONTINUE;
			}

			@Override
			public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
				String filename = file.getFileName().toString();

				if (!filename.startsWith(".") && filename.endsWith(".json")) {
					perf_metrics.add(file);
					System.out.println(filename);
				}
				return FileVisitResult.CONTINUE;
			}

		});

		final Path cwd_dir = Paths.get(System.getProperty("user.dir"));
		// git rev-parse HEAD
		String commit_id = exec_command(cwd_dir, "git", "rev-parse", "HEAD");
		String date = exec_command(cwd_dir, "git", "show", "-s", "--format=%ci", commit_id);
		final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z");
		java.util.Date commitDate = sdf.parse(date);
		final SimpleDateFormat simple_date_format = new SimpleDateFormat("yyyy-MM-dd");
		String batch_id = simple_date_format.format(commitDate);
		System.out.println(String.format("Commit change date: %s", batch_id));

		// collect all json files list
		processPerfMetrics(perf_metrics, commit_id, batch_id);

		// TODO - add e2e tests later, run it w/ process command
	}

	private static void processPerfMetrics(final List<Path> perf_metrics, String commit_id,
										   String batch_id) throws Exception {
		try {
			Connection conn = JdbcUtil.GetConn();
			System.out.println("MySQL DB connection established.\n");
			// go thru each json file
			JSONParser jsonParser = new JSONParser();
			for (Path metrics_json : perf_metrics) {
				try (FileReader reader = new FileReader(metrics_json.toAbsolutePath().toString())) {
					// Read JSON file
					Object obj = jsonParser.parse(reader);
					loadMetricsIntoMySQL(conn, commit_id, batch_id, (JSONObject) obj);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	static private void loadMetricsIntoMySQL(java.sql.Connection conn, String commit_id, String batch_id,
											 JSONObject json_object) throws Exception {

		try (java.sql.PreparedStatement st = conn.prepareStatement(
				"INSERT INTO perf_test_training_data (BatchId,CommitId,Model,ModelName,DisplayName,UseMixedPrecision,Optimizer,BatchSize,SeqLen,PredictionsPerSeq," +
						"NumOfBatches,WeightUpdateSteps,Round,GradAccSteps,AvgTimePerBatch,Throughput,StabilizedThroughput,TotalTime,AvgCPU,Memory,RunConfig,Time) " +
						"values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,Now())"
						+ "  ON DUPLICATE KEY UPDATE AvgTimePerBatch=?,Throughput=?,StabilizedThroughput=?,TotalTime=?,AvgCPU=?,Memory=?")) {

			int i = 0;

			// unique key section
			st.setString(++i, batch_id);
			st.setString(++i, commit_id.substring(0, 8));
			st.setString(++i, (String) json_object.get("Model"));
			st.setString(++i, (String) json_object.get("ModelName"));
			st.setString(++i, (String) json_object.get("DisplayName"));
			st.setBoolean(++i, (Boolean) json_object.get("UseMixedPrecision"));
			st.setString(++i, (String) json_object.get("Optimizer"));
			st.setInt(++i, (int)(long) json_object.get("BatchSize"));

			// non-key section
			JSONObject properties = (JSONObject) json_object.get("DerivedProperties");
			if (properties != null) {
				if (properties.get("SeqLen") == null) 				//  mysql allows null value in unique key column
					st.setNull(++i, Types.INTEGER);
				else
					st.setInt(++i, Integer.parseInt((String) properties.get("SeqLen")));

				if (properties.get("PredictionsPerSeq") == null) 				//  mysql allows null value in unique key column
					st.setNull(++i, Types.INTEGER);
				else
					st.setInt(++i, Integer.parseInt((String) properties.get("PredictionsPerSeq")));
			} else {
				st.setNull(++i, Types.INTEGER);
				st.setNull(++i, Types.INTEGER);
			}

			st.setInt(++i, (int)(long)  json_object.get("NumOfBatches"));
			st.setInt(++i, (int)(long)  json_object.get("WeightUpdateSteps"));
			st.setInt(++i, (int)(long)  json_object.get("Round"));
			st.setInt(++i, (int)(long)  json_object.get("GradAccSteps"));
			st.setFloat(++i, (float)(double) json_object.get("AvgTimePerBatch"));  // ms
			st.setFloat(++i, (float)(double) json_object.get("Throughput"));  // examples/sec
			st.setFloat(++i, (float)(double) json_object.get("StabilizedThroughput"));  // examples/sec
			st.setFloat(++i, (float)(double) json_object.get("TotalTime"));  // secs
			st.setInt(++i, (int)(long) json_object.get("AvgCPU"));
			st.setInt(++i, (int)((long) json_object.get("Memory") >> 20));  // mb
			st.setString(++i, (String) json_object.get("RunConfig"));

			// update section
			st.setFloat(++i, (float)(double) json_object.get("AvgTimePerBatch"));  // ms
			st.setFloat(++i, (float)(double) json_object.get("Throughput"));  // examples/sec
			st.setFloat(++i, (float)(double) json_object.get("StabilizedThroughput"));  // examples/sec
			st.setFloat(++i, (float)(double) json_object.get("TotalTime"));  // secs
			st.setInt(++i, (int)((long) json_object.get("Memory") >> 20));  // mb
			st.setString(++i, (String) json_object.get("RunConfig"));

			st.executeUpdate();
		} catch (Exception e) {
			e.printStackTrace();
			throw e;
		}

	}

}
