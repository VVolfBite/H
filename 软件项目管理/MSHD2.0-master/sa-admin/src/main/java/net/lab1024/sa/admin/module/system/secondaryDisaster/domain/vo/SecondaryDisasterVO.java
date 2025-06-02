package net.lab1024.sa.admin.module.system.secondaryDisaster.domain.vo;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.enumeration.SupportExtentEnum;
import net.lab1024.sa.common.common.enumeration.SupportLabelEnum;
import net.lab1024.sa.common.common.swagger.ApiModelPropertyEnum;

/**
 * ClassName: SecondaryDisasterVO
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.service.vo
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  20:50
 * @Version 1.0
 */
@Data
public class SecondaryDisasterVO {
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


    @ApiModelProperty("数据来源")
    private String Origin;

    @ApiModelProperty("数据载体")
    private String Carrier;

    @ApiModelProperty("是否删除0否 1是")
    private Boolean deletedFlag;
}
