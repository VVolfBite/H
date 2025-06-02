package sort;

import java.util.List;

/**
 * 排序模块的接口，定义排序方法。
 */
public interface SortStrategy {
    /**
     * 对行数据进行排序。
     * @param lines 包含需要排序的行
     * @return 排序后的行列表
     */
    List<String> sort(List<String> lines);
}
