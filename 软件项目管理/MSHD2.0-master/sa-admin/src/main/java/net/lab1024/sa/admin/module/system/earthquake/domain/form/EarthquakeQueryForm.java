package net.lab1024.sa.admin.module.system.earthquake.domain.form;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.domain.PageParam;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.Size;
import java.util.List;

@Data
public class EarthquakeQueryForm extends PageParam {
    @ApiModelProperty("搜索词")
    @Length(max = 40, message = "搜索词最多40字符")
    private String keyword;

    @ApiModelProperty("震情码集合")
    @Size(max = 99, message = "最多查询99条震情消息")
    private List<String> earthquakeCodeList;

    @ApiModelProperty(value = "删除标识", hidden = true)
    private Boolean deletedFlag;
}
