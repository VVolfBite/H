package sort;

import java.util.Collections;
import java.util.List;

/**
 * 默认排序实现，按字典序对行进行排序。
 */
public class DefaultSortStrategy implements SortStrategy {

    /**
     * 使用字典序对行数据进行排序。
     * @param lines 包含需要排序的行
     * @return 排序后的行列表
     */
    @Override
    public List<String> sort(List<String> lines) {
        Collections.sort(lines);
        return lines;
    }
}
