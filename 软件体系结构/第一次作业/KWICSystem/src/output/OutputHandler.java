package output;

import java.io.IOException;
import java.util.List;

/**
 * 输出处理接口，定义写入输出的方法。
 */
public interface OutputHandler {
    /**
     * 将行数据写入目标输出。
     * @param lines 包含所有需要输出的行
     * @throws IOException 如果写入过程中发生错误
     */
    void writeLines(List<String> lines) throws IOException;
}
