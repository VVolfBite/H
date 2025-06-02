package process;

import java.util.List;

/**
 * 数据处理模块的接口。
 */
public interface ProcessStrategy {
    /**
     * 根据输入行生成处理ProcessStrategy后的数据行。
     * @param line 输入行
     * @return 包含所有处理后的数据行
     */
    public List<String> processData(List<String> lines);
}
