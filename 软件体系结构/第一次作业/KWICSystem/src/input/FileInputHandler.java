package input;

import java.io.*;
import java.util.*;

/**
 * 从文件读取输入的实现类。
 */
public class FileInputHandler implements InputHandler {
    private final String filePath;

    /**
     * 构造函数
     * @param filePath 输入文件路径
     */
    public FileInputHandler(String filePath) {
        this.filePath = filePath;
    }

    /**
     * 从文件读取所有行并返回。
     * @return 包含所有文件行的列表
     * @throws IOException 如果读取文件时发生错误
     */
    @Override
    public List<String> readLines() throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    lines.add(line.trim());
                }
            }
        }
        return lines;
    }
}
