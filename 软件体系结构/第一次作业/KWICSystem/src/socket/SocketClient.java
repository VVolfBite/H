import java.io.*;
import java.net.*;
import java.util.Scanner;

public class SocketClient {
    private final String host;
    private final int port;

    /**
     * 构造函数
     * @param host 服务器地址
     * @param port 服务器端口号
     */
    public SocketClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    /**
     * 连接到服务器并发送数据。
     * @throws IOException 如果发生网络或写入错误
     */
    public void sendData() throws IOException {
        try (Socket socket = new Socket(host, port);
             PrintWriter writer = new PrintWriter(socket.getOutputStream(), true);
             Scanner scanner = new Scanner(System.in)) {

            System.out.println("Connected to server on port " + port);
            System.out.println("Enter data to send to server (press 'Ctrl+Z' to quit):");

            // 从控制台读取输入并发送到服务器
            String input;
            while (scanner.hasNextLine()) {
                input = scanner.nextLine();
                
                // 检查是否输入了 Ctrl+Z (EOF)
                if (input.equals("")) {
                    break;  // EOF 输入，退出
                }

                writer.println(input);  // 向服务器发送数据
            }
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java SocketClient <host> <port>");
            return;
        }

        String host = args[0];
        int port = Integer.parseInt(args[1]);

        SocketClient client = new SocketClient(host, port);
        try {
            client.sendData();
        } catch (IOException e) {
            System.err.println("Error occurred while sending data: " + e.getMessage());
        }
    }
}
