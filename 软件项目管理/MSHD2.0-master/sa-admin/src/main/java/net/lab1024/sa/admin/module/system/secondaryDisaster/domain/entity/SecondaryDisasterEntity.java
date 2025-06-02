package net.lab1024.sa.admin.module.system.secondaryDisaster.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * ClassName: SecondaryDisasterEntity
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.doamin.entity
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  20:45
 * @Version 1.0
 */
@Data
@TableName("t_secondary_disaster")
public class SecondaryDisasterEntity {
    /**
     * 统一编码
     */
    @TableId(type = IdType.INPUT)
    private String ID;

    /**
     * 发生的地点
     */
    private String Location;

    /**
     * 发生的时间
     */
    private String Time;

    /**
     * 受灾范围
     */
    private String SeverelyDamaged;

    /**
     * 受灾程度
     */
    private String Extent;

    /**
     * 描述
     */
    private String Description;

    /**
     * 受灾类型
     */
    private String Label;

    /**
     * 类型
     */
    private String Type;

    /**
     * 数据来源
     */
    private String Origin;

    /**
     * 载体形式
     */
    private String Carrier;

    /**
     * 是否删除0否 1是
     */
    private Boolean deletedFlag;

    /**
     * 备注
     */
    private String remark;

    private LocalDateTime updateTime;

    private LocalDateTime createTime;
}