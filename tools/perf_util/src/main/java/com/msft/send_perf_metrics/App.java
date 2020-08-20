package com.msft.send_perf_metrics;

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
import java.sql.PreparedStatement;
import java.text.SimpleDateFormat;
import java.util.*;

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

        // field name -> json value
        Map<String, Object> field_mapping = new LinkedHashMap();
        Set<String> update_on_duplicate_fields =
            new LinkedHashSet<> (Arrays.asList("AvgTimePerBatch", "Throughput", "StabilizedThroughput", "EndToEndThroughput", "TotalTime", "AvgCPU", "Memory"));

        field_mapping.put("BatchId", batch_id);
        field_mapping.put("CommitId", commit_id.substring(0, 8));
        json_object.forEach((key, value) -> {
            if (key.equals("DerivedProperties")) {
                JSONObject properties = (JSONObject) json_object.get("DerivedProperties");
                properties.forEach((sub_key, sub_value) -> {
                    field_mapping.put((String)sub_key, sub_value);
                });
            } else {
                field_mapping.put((String)key, value);
            }
        });

        // building sql statement
        StringBuilder sb = new StringBuilder("INSERT INTO perf_test_training_data (");
        field_mapping.forEach((key, value) -> {
            sb.append(key).append(",");
        });
        sb.append("Time) values (");
        for(int i = 0; i < field_mapping.size(); i++) {
            sb.append("?,");
        }
        sb.append("Now()) ON DUPLICATE KEY UPDATE ");
        update_on_duplicate_fields.forEach((key) -> {
            if(field_mapping.get(key) != null) {
                sb.append(key).append("=?,");
            }
        });

        try (java.sql.PreparedStatement st = conn.prepareStatement(sb.substring(0, sb.length() - 1))) {
            int i = 0; // param index
            for (Map.Entry<String, Object> entry : field_mapping.entrySet()) {
                setSqlParam(++i, st, entry.getValue());
            }

            // update section
            for(String key : update_on_duplicate_fields) {
                Object value = field_mapping.get(key);
                if(value != null) {
                    setSqlParam(++i, st, value);
                }
            }

            st.executeUpdate();
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }

    }

    static void setSqlParam(int param_index, PreparedStatement st, Object value) throws Exception {
        if (value instanceof String) {
            st.setString(param_index, (String) value);
        } else if (value instanceof Long) {
            st.setInt(param_index, (int) (long) value);
        } else if (value instanceof Double) {
            st.setFloat(param_index, (float) (double) value);
        } else if (value instanceof Boolean) {
            st.setBoolean(param_index, (Boolean) value);
        } else {
            throw new Exception("Unsupported data type:" + value.getClass().getName());
        }
    }

}
