package net.lab1024.sa.common.common.enumeration;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public enum CarrierEnum implements BaseEnum{
    /**
     * DIC
     */
    DIC(CarrierEnum.SystemEnvironmentNameConst.DIC, "文字"),

    /**
     * PIC
     */
    PIC(CarrierEnum.SystemEnvironmentNameConst.PIC, "图像"),

    /**
     * MEDIUM
     */
    AUD(CarrierEnum.SystemEnvironmentNameConst.AUD, "音频"),

    /**
     * VID
     */
    VID(CarrierEnum.SystemEnvironmentNameConst.VID, "视频"),

    /**
     * OTHERS
     */
    OTHERS(CarrierEnum.SystemEnvironmentNameConst.OTHERS, "其他");

    private final String value;

    private final String desc;

    public static final class SystemEnvironmentNameConst {
        public static final String DIC = "文字";
        public static final String PIC = "图像";
        public static final String AUD = "音频";
        public static final String VID = "视频";

        public static final String OTHERS = "其他";
    }
}
