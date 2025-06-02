package net.lab1024.sa.admin.module.system.secondaryDisaster.manager;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import net.lab1024.sa.admin.module.system.secondaryDisaster.dao.SecondaryDisasterDao;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.entity.SecondaryDisasterEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * ClassName: SecondaryDisasterManager
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.manager
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  21:04
 * @Version 1.0
 */
@Service
public class SecondaryDisasterManager extends ServiceImpl<SecondaryDisasterDao, SecondaryDisasterEntity> {
    @Autowired
    private SecondaryDisasterDao secondaryDisasterDao;


    /**
     * 保存生命线工程
     *
     * @param secondaryDisaster
     */
    @Transactional(rollbackFor = Throwable.class)
    public void saveSecondaryDisaster(SecondaryDisasterEntity secondaryDisaster) {
        // 保存员工 获得id
        secondaryDisasterDao.insert(secondaryDisaster);
    }

    /**
     * 更新生命线工程
     *
     * @param secondaryDisaster
     */
    @Transactional(rollbackFor = Throwable.class)
    public void updateSecondaryDisaster(SecondaryDisasterEntity secondaryDisaster) {
        // 保存员工 获得id
        secondaryDisasterDao.updateById(secondaryDisaster);
    }

}