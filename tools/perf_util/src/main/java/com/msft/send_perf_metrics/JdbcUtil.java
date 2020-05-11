package com.msft.send_perf_metrics;

import java.sql.DriverManager;
import java.util.Properties;

public class JdbcUtil {
	static java.sql.Connection GetConn() throws Exception {
		try (java.io.InputStream in = App.class.getResourceAsStream("/jdbc.properties")) {
			if (in == null)
				throw new RuntimeException("err");
			Properties props = new Properties();
			props.load(in);
			return DriverManager.getConnection(props.getProperty("url"), props.getProperty("user"),
					props.getProperty("password"));
		}
	}
}
