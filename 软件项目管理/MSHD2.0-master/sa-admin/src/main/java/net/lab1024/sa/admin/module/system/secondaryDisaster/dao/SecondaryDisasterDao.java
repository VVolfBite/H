package net.lab1024.sa.admin.module.system.secondaryDisaster.dao;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.entity.SecondaryDisasterEntity;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.form.SecondaryDisasterQueryForm;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.vo.SecondaryDisasterVO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.List;

/**
 * ClassName: SecondaryDisasterDao
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.dao
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  21:02
 * @Version 1.0
 */
@Mapper
@Component
public interface SecondaryDisasterDao extends BaseMapper<SecondaryDisasterEntity> {
    /**
     * 查询次生灾害列表
     *
     * @param page
     * @param queryForm
     * @return
     */
    List<SecondaryDisasterVO> querySecondaryDisaster(Page page, @Param("queryForm") SecondaryDisasterQueryForm queryForm);

    /**
     * 查询次生灾害
     */
    List<SecondaryDisasterVO> selectSecondaryDisasterByDeleted(@Param("deletedFlag") Boolean deletedFlag);

    /**
     * 通过ID查询
     *
     * @param ID
     * @return
     */
    SecondaryDisasterEntity getByID(@Param("ID") String ID);

    /**
     * 通过发生的地点查询
     *
     * @param location
     * @return
     */
    SecondaryDisasterEntity getByLocation(@Param("Location") String location);


    /**
     * 通过受灾程度查询
     *
     * @param Extent
     * @return
     */
    SecondaryDisasterEntity getByExtent(@Param("Extent") String Extent);

    /**
     * 通过受灾类型查询
     *
     * @param Label
     * @return
     */
    SecondaryDisasterEntity getByLabel(@Param("Label") String Label);

    /**
     * 获取所有员工
     *
     * @return
     */
    List<SecondaryDisasterVO> listAll();


    /**
     * 获取一批员工
     *
     * @param IDs
     * @return
     */
    List<SecondaryDisasterVO> getSecondaryDisasterByIds(@Param("IDs") Collection<String> IDs);


    /**
     * 查询单个员工信息
     *
     * @param ID
     * @return
     */
    SecondaryDisasterVO getSecondaryDisasterById(@Param("ID") String ID);

    /**
     * 获取所有
     *
     * @param leaveFlag
     * @return
     */
    List<String> getSecondaryDisasterId(@Param("leaveFlag") Boolean leaveFlag);
}

