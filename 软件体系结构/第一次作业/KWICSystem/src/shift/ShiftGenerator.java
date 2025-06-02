package shift;

import java.util.List;

/**
 * 循环位移生成模块的接口。
 */
public interface ShiftGenerator {
    /**
     * 根据输入行生成所有可能的循环位移。
     * @param line 输入行
     * @return 包含所有循环位移的列表
     */
    List<String> generateShifts(String line);
}
