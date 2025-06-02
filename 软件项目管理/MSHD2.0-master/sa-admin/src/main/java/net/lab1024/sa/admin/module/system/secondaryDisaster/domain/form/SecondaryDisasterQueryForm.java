package net.lab1024.sa.admin.module.system.secondaryDisaster.domain.form;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import net.lab1024.sa.common.common.domain.PageParam;
import org.hibernate.validator.constraints.Length;

import javax.validation.constraints.Size;
import java.util.List;

/**
 * ClassName: SecondaryDisasterQueryForm
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.doamin.form
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  20:54
 * @Version 1.0
 */
@Data
public class SecondaryDisasterQueryForm extends PageParam {
    @ApiModelProperty("搜索词")
    @Length(max = 40, message = "搜索词最多40字符")
    private String keyword;

    @ApiModelProperty("次生灾害id集合")
    @Size(max = 99, message = "最多查询99个次生灾害")
    private List<Long> IDList;

    @ApiModelProperty(value = "删除标识", hidden = true)
    private Boolean deletedFlag;
}