package net.lab1024.sa.admin.module.system.earthquake.domain.form;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.enumeration.CarrierEnum;
import net.lab1024.sa.common.common.util.SmartVerificationUtil;
import net.lab1024.sa.common.common.validator.enumeration.CheckEnum;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.NotNull;
import javax.validation.constraints.Pattern;
import javax.validation.constraints.Size;

@Data
public class EarthquakeAddForm {
    @ApiModelProperty("一体化编码")
    @NotNull(message = "编码不能为空")
    @Size(min = 36, max = 36, message = "编码为固定长度36位")
    private String Code;

    @ApiModelProperty("发生的地点")
    @Length(max = 100, message = "发生的地点最大长度为100")
    private String Location;

    @ApiModelProperty("发生的时间")
//    @Pattern(regexp = SmartVerificationUtil.DATE_TIME, message = "时间格式不正确")
    private String Datetime;

    @ApiModelProperty("数据来源大类")
    @Length(max = 50, message = "数据来源最大长度为50")
    private String Source;

    @ApiModelProperty("数据来源子类")
    @Length(max = 50, message = "数据来源子类最大长度为50")
    private String Subsource;

    @ApiModelProperty("载体")
    @CheckEnum(value = CarrierEnum.class, message = "载体错误")
    private String Carrier;

    @ApiModelProperty("灾情大类")
    @Length(max = 50, message = "灾情大类最大长度为50")
    private String Disaster;

    @ApiModelProperty("灾情子类")
    @Length(max = 50, message = "灾情子类最大长度为50")
    private String Subdisaster;

    @ApiModelProperty("灾情指标")
    @Length(max = 20, message = "灾情指标最大长度为20")
    private String Degree;

    @ApiModelProperty("备注")
    @Length(max = 200, message = "备注最大长度为200")
    private String Remark;

}
