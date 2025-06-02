package input;

import java.io.*;
import java.net.*;
import java.util.*;

/**
 * 从 Socket 读取输入的实现类。
 */
public class SocketInputHandler implements InputHandler {
    private final int port;

    /**
     * 构造函数
     * @param port 要监听的端口号
     */
    public SocketInputHandler(int port) {
        this.port = port;
    }

    /**
     * 从 Socket 读取所有行并返回。
     * @return 包含所有接收到的行的列表
     * @throws IOException 如果发生网络或读取错误
     */
    @Override
    public List<String> readLines() throws IOException {
        List<String> lines = new ArrayList<>();
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("Listening on port " + port + "...");
            try (Socket clientSocket = serverSocket.accept();
                 BufferedReader reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (!line.trim().isEmpty()) {
                        lines.add(line.trim());
                    }
                }
            }
        }
        return lines;
    }
}
