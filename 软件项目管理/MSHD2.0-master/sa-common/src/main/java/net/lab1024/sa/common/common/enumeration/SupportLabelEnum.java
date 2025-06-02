package net.lab1024.sa.common.common.enumeration;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public enum SupportLabelEnum implements BaseEnum {

    /**
     * TRANS
     */
    TRANS(SupportLabelEnum.SystemEnvironmentNameConst.TRANS, "交通"),

    /**
     * WATER
     */
    WATER(SupportLabelEnum.SystemEnvironmentNameConst.WATER, "供水"),

    /**
     * OIL
     */
    OIL(SupportLabelEnum.SystemEnvironmentNameConst.OIL, "输油"),

    /**
     * GAS
     */
    GAS(SupportLabelEnum.SystemEnvironmentNameConst.GAS, "燃气"),

    /**
     * ELECTRICITY
     */
    ELECTRICITY(SupportLabelEnum.SystemEnvironmentNameConst.ELECTRICITY, "电力"),

    /**
     * COMMUNICATION
     */
    COMMUNICATION (SupportLabelEnum.SystemEnvironmentNameConst.COMMUNICATION , "通信"),

    /**
     * HYDRAULIC
     */
    HYDRAULIC(SupportLabelEnum.SystemEnvironmentNameConst.HYDRAULIC, "水利");

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
