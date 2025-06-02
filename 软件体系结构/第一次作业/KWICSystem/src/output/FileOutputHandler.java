package output;

import java.io.*;
import java.util.List;

/**
 * 将输出写入文件的实现类。
 */
public class FileOutputHandler implements OutputHandler {
    private final String filePath;

    /**
     * 构造函数
     * @param filePath 输出文件路径
     */
    public FileOutputHandler(String filePath) {
        this.filePath = filePath;
    }

    /**
     * 将所有行写入指定的输出文件。
     * @param lines 包含所有需要输出的行
     * @throws IOException 如果写入文件时发生错误
     */
    @Override
    public void writeLines(List<String> lines) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String line : lines) {
                writer.write(line);
                writer.newLine();
            }
        }
        System.out.println("Output written to " + filePath);
    }
}
