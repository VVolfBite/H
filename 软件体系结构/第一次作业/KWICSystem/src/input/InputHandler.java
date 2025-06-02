package input;

import java.io.IOException;
import java.util.List;

/**
 * 输入处理接口，定义读取输入的方法。
 */
public interface InputHandler {
    /**
     * 读取输入数据并返回行列表。
     * @return 包含所有输入行的列表
     * @throws IOException 如果发生输入错误
     */
    List<String> readLines() throws IOException;
}
