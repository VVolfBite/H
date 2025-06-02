package net.lab1024.sa.admin.module.system.earthquake.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("t_earthquake")
public class EarthquakeEntity {
    @TableId(type = IdType.AUTO)
    private String code;

    /**
     * 发生的地点
     */
    private String location;

    /**
     * 发生的时间
     */
    private String datetime;

    /**
     * 数据来源大类
     */
    private String source;

    /**
     * 数据来源子类
     */
    private String subsource;

    /**
     * 载体
     */
    private String carrier;

    /**
     * 灾情大类
     */
    private String disaster;

    /**
     * 灾情子类
     */
    private String subdisaster;

    /**
     * 灾情指标
     */
    private String degree;

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
