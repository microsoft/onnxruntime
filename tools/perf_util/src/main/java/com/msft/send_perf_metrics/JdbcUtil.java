package com.msft.send_perf_metrics;

import java.sql.DriverManager;
import java.util.Properties;

public class JdbcUtil {
	static java.sql.Connection GetConn() throws Exception {
		try (java.io.InputStream in = App.class.getResourceAsStream("/jdbc.properties")) {
			if (in == null)
				throw new RuntimeException("Error reading jdbc properties");
			Properties props = new Properties();
			props.load(in);

			String password = System.getenv("ORT_PERF_PASSWORD");
			// ORT_PERF_PASSWORD

//			return DriverManager.getConnection(props.getProperty("url"), props.getProperty("user"),
//					props.getProperty("password"));

			return DriverManager.getConnection(props.getProperty("url"), props.getProperty("user"),
					password);
		}
	}
}
