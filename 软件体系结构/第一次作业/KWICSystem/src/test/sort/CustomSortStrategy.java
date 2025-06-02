package sort;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * 自定义排序实现，可以按用户指定的规则排序。
 */
public class CustomSortStrategy implements SortStrategy {
    private final Comparator<String> comparator;

    /**
     * 构造函数
     * @param comparator 自定义的比较器
     */
    public CustomSortStrategy(Comparator<String> comparator) {
        this.comparator = comparator;
    }

    /**
     * 使用自定义规则对行数据进行排序。
     * @param lines 包含需要排序的行
     * @return 排序后的行列表
     */
    @Override
    public List<String> sort(List<String> lines) {
        lines.sort(comparator);
        return lines;
    }
}
