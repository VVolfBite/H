package shift;

import java.util.ArrayList;
import java.util.List;

/**
 * 循环位移生成模块的默认实现类。
 */
public class CyclicShiftGenerator implements ShiftGenerator {

    /**
     * 根据输入行生成所有可能的循环位移。
     * @param line 输入行
     * @return 包含所有循环位移的列表
     */
    @Override
    public List<String> generateShifts(String line) {
        List<String> shifts = new ArrayList<>();
        if (line == null || line.isEmpty()) {
            return shifts;
        }

        String[] words = line.split("\\s+"); // 按空白字符分割
        int wordCount = words.length;

        for (int i = 0; i < wordCount; i++) {
            StringBuilder shiftedLine = new StringBuilder();
            for (int j = 0; j < wordCount; j++) {
                shiftedLine.append(words[(i + j) % wordCount]);
                if (j < wordCount - 1) {
                    shiftedLine.append(" ");
                }
            }
            shifts.add(shiftedLine.toString());
        }
        return shifts;
    }
}
