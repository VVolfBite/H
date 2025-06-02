package net.lab1024.sa.admin.module.system.CasualtyInfo.domain.vo;


import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.enumeration.CasualtyEnum;

import net.lab1024.sa.common.common.swagger.ApiModelPropertyEnum;

@Data
public class CasualtyVO {
    @ApiModelProperty("伤亡信息ID")
    private String ID;

    @ApiModelProperty("伤亡地点")
    private String Location;

    @ApiModelProperty("伤亡时间")
    private String Time;

    @ApiModelProperty("伤亡类型")
    @ApiModelPropertyEnum(CasualtyEnum.class)
    private String Label;

    @ApiModelProperty("伤亡及失踪数量")
    private Integer Number;

    @ApiModelProperty("数据来源")
    private String Origin;

    @ApiModelProperty("载体信息")
    private String Carrier;
}
