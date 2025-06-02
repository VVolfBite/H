package net.lab1024.sa.common.common.enumeration;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * ClassName: SecondaryLabelEnum
 * Package: net.lab1024.sa.common.common.enumeration
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  21:12
 * @Version 1.0
 */
@AllArgsConstructor
@Getter
public enum SecondaryLabelEnum implements BaseEnum {

    /**
     * TRANS
     */
    TRANS(SecondaryLabelEnum.SystemEnvironmentNameConst.TRANS, "交通"),

    /**
     * WATER
     */
    WATER(SecondaryLabelEnum.SystemEnvironmentNameConst.WATER, "供水"),

    /**
     * OIL
     */
    OIL(SecondaryLabelEnum.SystemEnvironmentNameConst.OIL, "输油"),

    /**
     * GAS
     */
    GAS(SecondaryLabelEnum.SystemEnvironmentNameConst.GAS, "燃气"),

    /**
     * ELECTRICITY
     */
    ELECTRICITY(SecondaryLabelEnum.SystemEnvironmentNameConst.ELECTRICITY, "电力"),

    /**
     * COMMUNICATION
     */
    COMMUNICATION (SecondaryLabelEnum.SystemEnvironmentNameConst.COMMUNICATION , "通信"),

    /**
     * HYDRAULIC
     */
    HYDRAULIC(SecondaryLabelEnum.SystemEnvironmentNameConst.HYDRAULIC, "水利");

    private final String value;

    private final String desc;

    public static final class SystemEnvironmentNameConst {
        public static final String TRANS = "交通";
        public static final String WATER = "供水";
        public static final String OIL = "输油";
        public static final String GAS = "燃气";
        public static final String ELECTRICITY = "电力";
        public static final String COMMUNICATION = "通信";
        public static final String HYDRAULIC = "水利";
    }
}