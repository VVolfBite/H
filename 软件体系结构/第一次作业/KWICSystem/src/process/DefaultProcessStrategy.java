package process;

import java.util.List;

/**
 * 默认数据处理，输入行原路返回。
 */
public class DefaultProcessStrategy implements ProcessStrategy {

    /**
     * 根据输入行生成处理后的数据行。
     * @param lines 包含需要处理的行
     * @return 包含所有处理后的数据行
     */
    @Override
    public List<String> processData(List<String> lines) {
        return lines;
    }
}
