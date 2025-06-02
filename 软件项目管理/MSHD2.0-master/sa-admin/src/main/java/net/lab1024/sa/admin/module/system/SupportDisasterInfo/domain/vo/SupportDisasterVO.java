package net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.enumeration.SupportExtentEnum;
import net.lab1024.sa.common.common.enumeration.SupportLabelEnum;
import net.lab1024.sa.common.common.swagger.ApiModelPropertyEnum;

/**
 * 生命线工程信息
 *
 * @Author 1024创新实验室: 罗伊
 * @Date 2021-12-21 23:05:56
 * @Wechat zhuoda1024
 * @Email lab1024@163.com
 * @Copyright 1024创新实验室 （ https://1024lab.net ）
 */
@Data
public class SupportDisasterVO {
    @ApiModelProperty("主键ID")
    private String ID;

    @ApiModelProperty("发生的地点")
    private String location;

    @ApiModelProperty("发生的时间")
    private String Time;

    @ApiModelProperty("受灾范围")
    private String SeverelyDamaged;

    @ApiModelPropertyEnum(SupportExtentEnum.class)
    private String Extent;

    @ApiModelProperty("描述")
    private String Description;

    @ApiModelPropertyEnum(SupportLabelEnum.class)
    private String Label;

    @ApiModelProperty("类型")
    private String Type;

    @ApiModelProperty("受灾设施数")
    private Integer Number;

    @ApiModelProperty("数据来源")
    private String Origin;

    @ApiModelProperty("数据载体")
    private String Carrier;

    @ApiModelProperty("是否删除0否 1是")
    private Boolean deletedFlag;
}
