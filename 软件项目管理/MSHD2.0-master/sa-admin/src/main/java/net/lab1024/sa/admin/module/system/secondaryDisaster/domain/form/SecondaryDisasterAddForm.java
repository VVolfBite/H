package net.lab1024.sa.admin.module.system.secondaryDisaster.domain.form;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.enumeration.CarrierEnum;
import net.lab1024.sa.common.common.enumeration.SupportExtentEnum;
import net.lab1024.sa.common.common.enumeration.SupportLabelEnum;
import net.lab1024.sa.common.common.util.SmartVerificationUtil;
import net.lab1024.sa.common.common.validator.enumeration.CheckEnum;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotNull;
import javax.validation.constraints.Pattern;
import javax.validation.constraints.Size;

/**
 * ClassName: SecondaryDisasterAddForm
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.doamin.form
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  20:53
 * @Version 1.0
 */
@Data
public class SecondaryDisasterAddForm {
    @ApiModelProperty("ID")
    @NotNull(message = "编码不能为空")
    @Size(min = 36, max = 36, message = "编码为固定长度36位")
    private String ID;

    @ApiModelProperty("发生的地点")
    @NotNull(message = "发生的地点不能为空")
    @Length(max = 100, message = "发生的地点最大长度为100")
    private String location;

    @ApiModelProperty("发生的时间")
    @NotNull(message = "发生的时间不能为空")
    @Pattern(regexp = SmartVerificationUtil.DATE_TIME, message = "时间格式不正确")
    private String Time;

    @ApiModelProperty("受灾范围")
    @NotNull(message = "受灾范围不能为空")
    @Length(max = 50, message = "受灾范围最大长度为50")
    private String SeverelyDamaged;

    @ApiModelProperty("受灾程度")
    @NotNull(message = "受灾程度不能为空")
    @CheckEnum(value = SupportExtentEnum.class, message = "受灾程度错误")
    private String Extent;

    @ApiModelProperty("描述")
    @NotNull(message = "描述不能为空")
    @Length(max = 200, message = "描述最大长度为200")
    private String Description;

    @ApiModelProperty("受灾类型")
    @NotNull(message = "受灾类型不能为空")
    @CheckEnum(value = SupportLabelEnum.class, message = "受灾类型错误")
    private String Label;

    @ApiModelProperty("类型")
    @NotNull(message = "类型不能为空")
    @Length(max = 4, message = "类型最大长度为4")
    private String Type;


    @ApiModelProperty("数据来源")
    @NotNull(message = "数据来源不能为空")
    @Length(max = 50, message = "数据来源最大长度为50")
    private String Origin;

    @ApiModelProperty("数据载体")
    @NotNull(message = "数据载体不能为空")
    @CheckEnum(value = CarrierEnum.class, message = "数据载体错误")
    private String Carrier;
}
