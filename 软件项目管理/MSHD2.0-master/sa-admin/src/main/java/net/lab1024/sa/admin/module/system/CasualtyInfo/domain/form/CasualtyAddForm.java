package net.lab1024.sa.admin.module.system.CasualtyInfo.domain.form;

import io.swagger.annotations.ApiModelProperty;

import lombok.Data;
import net.lab1024.sa.common.common.enumeration.CarrierEnum;
import net.lab1024.sa.common.common.enumeration.CasualtyEnum;
import net.lab1024.sa.common.common.validator.enumeration.CheckEnum;
import org.hibernate.validator.constraints.Length;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;

@Data
public class CasualtyAddForm {
    @ApiModelProperty("ID")
    @NotNull(message = "编码不可为空")
    @Size(min = 36,max =36,message ="编码固定长度36位")
    private String ID;

    @ApiModelProperty("人员伤亡地点")
    @NotNull(message = "人员伤亡地址不能为空")
    @Length(max = 100,message = "人员伤亡的地点不能为空")
    private String Location;

    @ApiModelProperty("人员伤亡的时间")
    @NotNull(message = "人员伤亡的时间不能为空")
    @Length(max = 14,message = "人员伤亡的时间范围最大为14")
    private String Time;

    @ApiModelProperty("伤亡类型")
    @NotNull(message = "人员伤亡的类型不能为空")
    @CheckEnum(value = CasualtyEnum.class,message = "伤亡类型错误")
    private String Label;

    @ApiModelProperty("伤亡数量")
    @NotNull(message = "人员伤亡的数量不能为空")
    @Length(max = 5,message = "伤亡数量的范围最大为5")
    private Integer Number;

    @ApiModelProperty("数据来源")
    @NotNull(message = "数据的来源不能为空")
    @Length(max = 50,message = "数据来源的范围最大为50")
    private String Origin;

    @ApiModelProperty("载体")
    @NotNull(message = "载体不能为空")
    @CheckEnum(value = CarrierEnum.class, message = "受灾程度错误")
    private String Carrier;
}
