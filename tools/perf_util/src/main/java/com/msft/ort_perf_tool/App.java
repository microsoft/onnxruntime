package com.msft.ort_perf_tool;

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
				}
				return FileVisitResult.CONTINUE;
			}

		});
		System.out.println(perf_metrics);

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
		processPerfMetrics(source_dir, perf_metrics, commit_id, batch_id);

		// TODO - add e2e tests later, run it w/ process command
	}

	private static void processPerfMetrics(Path source_dir, final List<Path> perf_metrics, String commit_id,
										   String batch_id) throws Exception {
		try {
			Connection conn = JdbcUtil.GetConn();
			// go thru each json file
			JSONParser jsonParser = new JSONParser();
			for (Path metrics_json : perf_metrics) {
				try (FileReader reader = new FileReader(metrics_json.toAbsolutePath().toString())) {
					// Read JSON file
					Object obj = jsonParser.parse(reader);
					loadMetricsIntoMySQL(conn, commit_id, batch_id, (JSONObject) obj);
				}
			}
		} finally {
			deletePath(source_dir);
		}
	}

	private static void deletePath(Path temp_path) throws IOException {
		Files.walkFileTree(temp_path, new SimpleFileVisitor<Path>() {
			@Override
			public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
				String filename = file.getFileName().toString();
				if (!filename.startsWith(".") && filename.endsWith(".json")) {
					Files.deleteIfExists(file);
				}

				Files.deleteIfExists(file);
				return FileVisitResult.CONTINUE;
			}
		});
	}

	static private void loadMetricsIntoMySQL(java.sql.Connection conn, String commit_id, String batch_id,
											 JSONObject json_object) throws Exception {

		try (java.sql.PreparedStatement st = conn.prepareStatement(
				"INSERT INTO perf_test_training_data (BatchId,CommitId,Model,UsedPrecision,Optimizer,BatchSize,SeqLen,PredictionsPerSeq," +
						"NumOfBatches,WeightUpdateSteps,Round,GradAccSteps,AvgTimePerBatch,Throughput,TotalTime,AvgCPU,Memory,RunConfig,Time) " +
						"values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,Now())"
						+ "  ON DUPLICATE KEY UPDATE AvgTimePerBatch=?,Throughput=?,TotalTime=?,AvgCPU=?,Memory=?")) {

			// unique key section
			st.setString(1, batch_id);
			st.setString(2, commit_id.substring(0, 8));
			st.setString(3, (String) json_object.get("Model"));
			st.setString(4, (String) json_object.get("Precision"));
			st.setString(5, (String) json_object.get("Optimizer"));
			st.setInt(6, (int)(long) json_object.get("BatchSize"));
			if (json_object.get("SeqLen") == null) 				//  mysql allows null value in unique key column
				st.setNull(7, Types.INTEGER);
			else
				st.setFloat(7, (int)(long) json_object.get("SeqLen"));

			// non-key section
			if (json_object.get("PredictionsPerSeq") == null)
				st.setNull(8, Types.INTEGER);
			else
				st.setFloat(8, (int)(long) json_object.get("PredictionsPerSeq"));
			st.setInt(9, (int)(long)  json_object.get("NumOfBatches"));
			st.setInt(10, (int)(long)  json_object.get("WeightUpdateSteps"));
			st.setInt(11, (int)(long)  json_object.get("Round"));
			st.setInt(12, (int)(long)  json_object.get("GradAccSteps"));
			st.setFloat(13, (float)(double) json_object.get("AvgTimePerBatch"));  // ms
			st.setFloat(14, (float)(double) json_object.get("Throughput"));  // examples/sec
			st.setFloat(15, (float)(double) json_object.get("TotalTime"));  // secs
			// TODO - remove "if" check
			if (json_object.get("AvgCPU") == null)
				st.setNull(16, Types.FLOAT);
			else
				st.setFloat(16, (float)(double) json_object.get("AvgCPU"));

			if (json_object.get("Memory") == null)
				st.setNull(17, Types.INTEGER);
			else
				st.setInt(17, (int)(long) json_object.get("Memory"));  // mb

			st.setString(18, (String) json_object.get("RunConfig"));

			// update section
			st.setFloat(19, (float)(double) json_object.get("AvgTimePerBatch"));  // ms
			st.setFloat(20, (float)(double) json_object.get("Throughput"));  // examples/sec
			st.setFloat(21, (float)(double) json_object.get("TotalTime"));  // secs
			if (json_object.get("AvgCPU") == null)
				st.setNull(22, Types.FLOAT);
			else
				st.setFloat(22, (float)(double) json_object.get("AvgCPU"));

			if (json_object.get("Memory") == null)
				st.setNull(23, Types.INTEGER);
			else
				st.setInt(23, (int)(long) json_object.get("Memory"));  // mb

			st.executeUpdate();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
