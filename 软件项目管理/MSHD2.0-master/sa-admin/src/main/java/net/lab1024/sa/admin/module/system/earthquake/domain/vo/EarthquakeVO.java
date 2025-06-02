package net.lab1024.sa.admin.module.system.earthquake.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.enumeration.CarrierEnum;
import net.lab1024.sa.common.common.swagger.ApiModelPropertyEnum;

/**
 * 震情信息
 *
 * @Author 1024创新实验室: 罗伊
 * @Date 2021-12-21 23:05:56
 * @Wechat zhuoda1024
 * @Email lab1024@163.com
 * @Copyright 1024创新实验室 （ https://1024lab.net ）
 */
@Data
public class EarthquakeVO {
    @ApiModelProperty("主键ID")
    String code;

    @ApiModelProperty("地理信息")
    String location;

    @ApiModelProperty("时间")
    String datetime;

    @ApiModelProperty("数据来源")
    String source;

    @ApiModelProperty("数据来源子类")
    String subsource;

    @ApiModelPropertyEnum(CarrierEnum.class)
    String carrier;

    @ApiModelProperty("灾情大类")
    String disaster;

    @ApiModelProperty("灾情子类")
    String subdisaster;

    @ApiModelProperty("灾情指标")
    String degree;

    @ApiModelProperty("备注")
    String remark;
}
