package net.lab1024.sa.common.common.enumeration;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * ClassName: SecondaryExtentEnum
 * Package: net.lab1024.sa.common.common.enumeration
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  21:11
 * @Version 1.0
 */
@AllArgsConstructor
@Getter
public enum SecondaryExtentEnum implements BaseEnum{

    /**
     * WELL
     */
    WELL(SecondaryExtentEnum.SystemEnvironmentNameConst.WELL, "基本完好"),

    /**
     * SLIGHTLY
     */
    SLIGHTLY(SecondaryExtentEnum.SystemEnvironmentNameConst.SLIGHTLY, "轻微破坏"),

    /**
     * MEDIUM
     */
    MEDIUM(SecondaryExtentEnum.SystemEnvironmentNameConst.MEDIUM, "中等破坏"),

    /**
     * HEAVY
     */
    HEAVY(SecondaryExtentEnum.SystemEnvironmentNameConst.HEAVY, "严重破坏"),

    /**
     * RUIN
     */
    RUIN(SecondaryExtentEnum.SystemEnvironmentNameConst.RUIN, "毁坏");

    private final String value;

    private final String desc;

    public static final class SystemEnvironmentNameConst {
        public static final String WELL = "基本完好";
        public static final String SLIGHTLY = "轻微破坏";
        public static final String MEDIUM = "中等破坏";
        public static final String HEAVY = "严重破坏";

        public static final String RUIN = "毁坏";
    }
}
